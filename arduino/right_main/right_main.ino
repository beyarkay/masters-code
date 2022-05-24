#include <Wire.h>
const int NUM_SENSORS = 15;
const int NUM_SELECT_PINS = 4;
const int PINS_SENSOR_SELECT[] = {8, 9, 10, 11};
const int PIN_SENSOR_INPUT = A0;
int val = 0;
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

// left hand can send up to ((4 + 1) * 15) = 75 bytes of data at a time
const int left_hand_cap = 75;
int left_hand_len = 0;
char left_hand[75];
boolean new_data = false;

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
    Serial.begin(57600);
}

void loop() {
    // If we've got a data packet from the left hand
    if (left_hand_len > 0) {
        // First calculate and print out the gesture index.
        // Read in the DIP switches to figure out what gesture index we're at
        // Init `gesture_index` to zero so bit shifts can work properly
        gesture_index = 0x00;
        for (int i = 0; i < NUM_DIP_SWITCHES; i++) {
            int val = 1 - digitalRead(PINS_DIP_SWITCHES[i]);
            gesture_index |= val << (NUM_DIP_SWITCHES - i - 1);
        }
        // Print the gesture_index
        Serial.print(gesture_index);
        Serial.print(",");

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
        // Print out the time since the start of the gesture
        Serial.print(millis() - last_beep);
        Serial.print(",");

        // Then print out the data from the left hand
        for (int i = 0; i < left_hand_len; i++) {
            Serial.print(left_hand[i]);
        }
        left_hand_len = 0;

        // Finally print out the data from the right hand
        for (int i = 0; i < NUM_SENSORS; i++) {
            for (int j = 0; j < NUM_SELECT_PINS; j++) {
                digitalWrite(PINS_SENSOR_SELECT[j], (i & (1 << j)) >> j);
            }
            val = analogRead(PIN_SENSOR_INPUT);
            // Serial.print(val);
            int thou = val / 1000;
            if (thou > 0) {
                Serial.write('0' + thou);
                val -= thou * 1000;
            }
            int hund = val / 100;
            if (hund > 0 || thou) {
                Serial.write('0' + hund);
                val -= hund * 100;
            }
            int tens = val / 10;
            if (tens > 0 || hund || thou) {
                Serial.write('0' + tens);
                val -= tens * 10;
            }
            if (val > 0 || tens || hund || thou) {
                Serial.write('0' + val);
            }
            Serial.print(',');
        }

        // And end it all off with a newline
        Serial.print("\n");
    }
}

void receiveEvent(int num_events) {
    digitalWrite(LED_BUILTIN, HIGH);
    left_hand_len = 0;
    while (Wire.available() > 0) {
        left_hand[left_hand_len++] = Wire.read();
    }
    delay(1);
    digitalWrite(LED_BUILTIN, LOW);
}

