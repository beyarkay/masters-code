#include <Wire.h>
const int NUM_SENSORS = 15;
const int NUM_SELECT_PINS = 4;
const int PINS_SENSOR_SELECT[] = {8, 9, 10, 11};
const int PIN_SENSOR_INPUT = A0;
int vals_rh[] = {
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0
};
// There are 8 DIP switches that are used to select a gesture index
const int PINS_DIP_SWITCHES[] = {
    A3, A2, 7, 6,
    A6, A7, 4, 5
};
const int NUM_DIP_SWITCHES = 8;
// There are 255 indexable gestures 0x01 to 0xFF, with 0x00 reserved to
// indicate that the glove is in KEYBOARD mode and not in TRAINING mode.
int gesture_index = 0x01;
// The piezo-electric buzzer is used to indicate the boundary between training
// packets
const int PIN_BUZZER = 2;
int buzzer_state = 0;
const int MS_PER_TRAINING_PACKET = 1000;
const int MS_PER_BEEP = 30;
long last_beep = 0;

// Keep track of the last time we wrote to the serial port
long last_write = 0;
const int MIN_MS_PER_WRITE = 20;

// left hand can send up to ((4 + 1) * 15) = 75 bytes of data at a time
const int left_hand_cap = 75;
int left_hand_len = 0;
char left_hand[75];

const int right_hand_cap = 75;
int right_hand_len = 0;
char right_hand[75];

// alpha is used for exponential smoothing. High alpha => very smooth. This
// value is found experimentally.
const float alpha = 0.99;

void setup() {
    // Mark the builtin LED as output
    pinMode(LED_BUILTIN, OUTPUT);
    // Start up I2C communication with device on channel 4 (left hand)
    Wire.begin(4);
    // Register a function to handle received messages
    Wire.onReceive(receiveEvent);
    // Use a high baud rate so that the sensor readings are more accurate.
    Serial.begin(19200);
    // Mark the sensor input pin as an input
    pinMode(PIN_SENSOR_INPUT, INPUT);
    // Mark the sensor selection pins as output
    for (int j = 0; j < NUM_SELECT_PINS; j++) {
        pinMode(PINS_SENSOR_SELECT[j], OUTPUT);
    }
    // Mark the DIP switches as input pins which default to HIGH
    for (int i = 0; i < NUM_DIP_SWITCHES; i++) {
        pinMode(PINS_DIP_SWITCHES[i], INPUT_PULLUP);
    }
    // Mark the buzzer as an output
    pinMode(PIN_BUZZER, OUTPUT);
    // Set the last write time to now
    last_write = millis();
}

void loop() {
    if (millis() - last_write >= MIN_MS_PER_WRITE 
        && left_hand_len > 0 
        && right_hand_len > 0
    ) {
        last_write = millis();
        Serial.print(gesture_index);
        Serial.print(",");
        Serial.print(millis() - last_beep);
        Serial.print(",");
        // Print out the data from the left hand
        for (int i = 0; i < left_hand_len; i++) {
            Serial.write(left_hand[i]);
        }
        left_hand_len = 0;

        // Print out the data from the right hand
        for (int i = 0; i < right_hand_len; i++) {
            Serial.write(right_hand[i]);
        }
        right_hand_len = 0;
        Serial.print("\n");
    }
    // Calculate and print out the gesture index. Read in the DIP switches
    // to figure out what gesture index we're at Init `gesture_index` to
    // zero so bit shifts can work properly
    gesture_index = 0x00;
    for (int i = 0; i < NUM_DIP_SWITCHES; i++) {
        int gesture_val = 1 - digitalRead(PINS_DIP_SWITCHES[i]);
        gesture_index |= gesture_val << (NUM_DIP_SWITCHES - i - 1);
    }

    // Only sound the buzzer if we're in TRAINING mode and it's the correct
    // time interval
    if (gesture_index != 0
            && millis() % MS_PER_TRAINING_PACKET <= MS_PER_BEEP
            && buzzer_state != 150) {
        // sound the buzzer
        buzzer_state = 150;
        analogWrite(PIN_BUZZER, buzzer_state);
        last_beep = millis();
    } else if (buzzer_state != 0) {
        // Reset the buzer
        buzzer_state = 0;
        analogWrite(PIN_BUZZER, buzzer_state);
    }
    if (gesture_index == 0) {
        last_beep = millis();
    }

    right_hand_len = 0;
    for (int i = 0; i < NUM_SENSORS; i++) {
        for (int j = 0; j < NUM_SELECT_PINS; j++) {
            digitalWrite(PINS_SENSOR_SELECT[j], (i & (1 << j)) >> j);
        }
        vals_rh[i] = floor((1.0 - alpha) * analogRead(PIN_SENSOR_INPUT) + alpha * vals_rh[i]);

        int thou = vals_rh[i] / 1000;
        if (thou > 0) {
            right_hand[right_hand_len++] = '0' + thou;
            vals_rh[i] -= thou * 1000;
        }
        int hund = vals_rh[i] / 100;
        if (hund > 0 || thou) {
            right_hand[right_hand_len++] = '0' + hund;
            vals_rh[i] -= hund * 100;
        }
        int tens = vals_rh[i] / 10;
        if (tens > 0 || hund || thou) {
            right_hand[right_hand_len++] = '0' + tens;
            vals_rh[i] -= tens * 10;
        }
        if (vals_rh[i] > 0 || tens || hund || thou) {
            right_hand[right_hand_len++] = '0' + vals_rh[i];
        }
        vals_rh[i] += thou * 1000 + hund * 100 + tens * 10;
        right_hand[right_hand_len++] = ',';
    }
}

/*
 * Read incoming I2C data from the left hand.
 */
void receiveEvent(int num_events) {
    digitalWrite(LED_BUILTIN, HIGH);
    left_hand_len = 0;
    while (Wire.available() > 0) {
        left_hand[left_hand_len++] = Wire.read();
    }
    digitalWrite(LED_BUILTIN, LOW);
}

