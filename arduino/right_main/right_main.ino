#include <Wire.h>
const int NUM_SENSORS = 15;
const int NUM_SELECT_PINS = 4;
const int PINS_SENSOR_SELECT[] = {8, 9, 10, 11};
const int PIN_SENSOR_INPUT = A0;
// There are hardware differences in all the sensors. Add an offset to
// approximately remove these differences
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
const int MS_PER_TRAINING_PACKET = 500;
const int MS_PER_BEEP = 40;
const int START_TIME = (MS_PER_TRAINING_PACKET - MS_PER_BEEP) / 2;
const int FINSH_TIME = (MS_PER_TRAINING_PACKET + MS_PER_BEEP) / 2;

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
const float alpha = 0.00;

void setup() {
    // Mark the builtin LED as output
    pinMode(LED_BUILTIN, OUTPUT);
    // Start up I2C communication with device on channel 42 (left hand)
    Wire.begin(42);
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
        && right_hand_len > 0) {
        last_write = millis();
        Serial.print(gesture_index);
        Serial.print(",");
        Serial.print(gesture_index == 0 ? 0 : millis() % MS_PER_TRAINING_PACKET);
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
    int time_offset = millis() % MS_PER_TRAINING_PACKET;
    if (gesture_index != 0
            && START_TIME <= time_offset
            && time_offset <= FINSH_TIME
            && buzzer_state == 0
    ) {
        // sound the buzzer at 110Hz (A2)
        // Gesture index is in [1,255], multiply by 8 to get [8,2040], add 50
        // to get in human hearing range of [58,2090]
        buzzer_state = gesture_index << 3 + 50;
        tone(PIN_BUZZER, buzzer_state);
    }
    if (
        (time_offset < START_TIME || FINSH_TIME < time_offset)
        && buzzer_state != 0
    ) {
        // Reset the buzer
        buzzer_state = 0;
        noTone(PIN_BUZZER);
    }

    right_hand_len = 0;
    for (int zyx = 0; zyx < NUM_SENSORS; zyx++) {
        // There's an issue where the order of the dimensions for the right
        // hand is ZYX not XYZ. This expression fixes that. 0=>2, 1=>1, 2=>0,
        // 3=>5, 4=>4, 5=>3, and so on.
        int offset = (floor(zyx/3) * 3 + 1);
        int xyz = - (zyx - offset) + offset;
        for (int j = 0; j < NUM_SELECT_PINS; j++) {
            digitalWrite(PINS_SENSOR_SELECT[j], (xyz & (1 << j)) >> j);
        }
        int reading = analogRead(PIN_SENSOR_INPUT);
        vals_rh[xyz] = floor((1.0 - alpha) * reading + alpha * vals_rh[xyz]);

        int thou = vals_rh[xyz] / 1000;
        if (thou > 0) {
            right_hand[right_hand_len++] = '0' + thou;
            vals_rh[xyz] -= thou * 1000;
        }
        int hund = vals_rh[xyz] / 100;
        if (hund > 0 || thou) {
            right_hand[right_hand_len++] = '0' + hund;
            vals_rh[xyz] -= hund * 100;
        }
        int tens = vals_rh[xyz] / 10;
        if (tens > 0 || hund || thou) {
            right_hand[right_hand_len++] = '0' + tens;
            vals_rh[xyz] -= tens * 10;
        }
        if (vals_rh[xyz] > 0 || tens || hund || thou) {
            right_hand[right_hand_len++] = '0' + vals_rh[xyz];
        }
        vals_rh[xyz] += thou * 1000 + hund * 100 + tens * 10;
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

