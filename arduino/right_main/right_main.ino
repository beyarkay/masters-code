#include <Wire.h>
const int NUM_SENSORS = 15;
const int NUM_SELECT_PINS = 4;
const int PINS_SENSOR_SELECT[] = {8, 9, 10, 11};
const int PIN_SENSOR_INPUT = A0;
int val = 0;
// // There are 8 DIP switches that are used to select a gesture index
// const int PINS_DIP_SWITCHES[] = {
//      A3, A2, 7, 6,
//      A6, A7, 4, 5
// };
// const int NUM_DIP_SWITCHES = 8;
// There are 255 indexable gestures 0x01 to 0xFF, with 0x00 reserved to
// indicate that the glove is in KEYBOARD mode and not in TRAINING mode.
// int gesture_index = 0x01;
// The piezo-electric buzzer is used to indicate the boundary between training
// packets
// const int PIN_BUZZER = 2;
// int buzzer_state = 0;
// int must_deliminate_packet = 0;
// const int ms_per_training_packet = 1000;
// const int ms_per_beep = 50;
// long last_beep = 0;

// left hand can send up to ((4 + 1) * 15) = 75 bytes of data at a time
int left_hand_len = 75;
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
    Serial.begin(57600);
    // Mark the sensor input pin as an input
    pinMode(PIN_SENSOR_INPUT, INPUT);
    // Mark the sensor selection pins as output
    for (int j = 0; j < NUM_SELECT_PINS; j++) {
        pinMode(PINS_SENSOR_SELECT[j], OUTPUT);
    }
    // Mark the DIP switches as input pins which default to HIGH
    // for (int i = 0; i < NUM_DIP_SWITCHES; i++) {
    //     pinMode(PINS_DIP_SWITCHES[i], INPUT_PULLUP);
    // }
    // // Mark the buzzer as an output
    // pinMode(PIN_BUZZER, OUTPUT);
    Serial.begin(57600);
}

void loop() {
    // If we've got a data packet from the left hand
    if (new_data) {
        for (int i = 0; i < left_hand_len; i++) {
            Serial.print(left_hand[i]);
        }
        new_data = false;
        for (int i = 0; i < NUM_SENSORS; i++) {
            for (int j = 0; j < NUM_SELECT_PINS; j++) {
                digitalWrite(PINS_SENSOR_SELECT[j], (i & (1 << j)) >> j);
            }
            val = analogRead(PIN_SENSOR_INPUT);
            char buffer[5];
            sprintf(buffer, "%4i,", val);
            Serial.print(buffer);
            Serial.print(",");
        }
        Serial.print("\n");
    }
}

void receiveEvent(int num_events) {
    digitalWrite(LED_BUILTIN, HIGH);
    int idx = 0;
    while (Wire.available() > 0) {
        left_hand[idx++] = Wire.read();
    }
    while (idx < left_hand_len) {
        left_hand[idx++] = '_';
    }
    new_data = true;
    delay(10);
    digitalWrite(LED_BUILTIN, LOW);
}

