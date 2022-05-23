const int NUM_SENSORS = 15;
const int PIN_SENSOR_SELECT_0 = 8;
const int PIN_SENSOR_SELECT_1 = 9;
const int PIN_SENSOR_SELECT_2 = 10;
const int PIN_SENSOR_SELECT_3 = 11;
const int PINS_SENSOR_SELECT[] = {8, 9, 10, 11};
const int NUM_SELECT_PINS = 4;
const int PIN_SENSOR_INPUT = A0;
const int MS_PER_ANALOG_READ = 1;
// There are 8 DIP switches that are used to select a gesture index
// FIXME this is wrong
const int PINS_DIP_SWITCHES[] = {
     A3, A2, 7, 6,
     A4, A5, 4, 5
};
const int NUM_DIP_SWITCHES = 8;
// There are 255 indexable gestures 0x01 to 0xFF, with 0x00 reserved to
// indicate that the glove is in KEYBOARD mode and not in TRAINING mode.
int gesture_index = 0x01;
// The piezo-electric buzzer is used to indicate the boundary between training
// packets
const int PIN_BUZZER = 2;
int buzzer_state = 0;

int must_deliminate_packet = 0;

const int ms_per_training_packet = 1000;
const int ms_per_beep = 50;

long last_beep = 0;

void setup() {
    // Use a high baud rate so that the sensor readings are more accurate.
    Serial.begin(57600);
    pinMode(PIN_SENSOR_INPUT, INPUT);
    for (int j = 0; j < NUM_SELECT_PINS; j++) {
        pinMode(PINS_SENSOR_SELECT[j], OUTPUT);
    }
    for (int i = 0; i < NUM_DIP_SWITCHES; i++) {
        pinMode(PINS_DIP_SWITCHES[i], INPUT_PULLUP);
    }
    // Buzzer
    pinMode(PIN_BUZZER, OUTPUT);
}

void loop() {
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
            && millis() % ms_per_training_packet <= ms_per_beep
            && buzzer_state != 150) {
        buzzer_state = 150;
        analogWrite(PIN_BUZZER, buzzer_state);
        last_beep = millis();
    } else if (buzzer_state != 0) {
        buzzer_state = 0;
        analogWrite(PIN_BUZZER, buzzer_state);
    }
    if (gesture_index == 0) {
        last_beep = millis();
    }
    // Write the millis since the last beep, for training purposes
    Serial.print(millis() - last_beep);
    Serial.print(",");

    // Only send a training packet deliminator if we're in TRAINING mode and
    // about `ms_per_training_packet` has elapsed and we haven't already sent a
    // training packet deliminator
    if (gesture_index != 0
            && millis() % ms_per_training_packet < 20
            && must_deliminate_packet == 1) {
        for (int i = 0; i < NUM_SENSORS; i++) {
            Serial.print("0,");
        }
        must_deliminate_packet = 0;
    } else {
        // Print each sensor value from the multiplexor
        for (int i = 0; i < NUM_SENSORS; i++) {
            for (int j = 0; j < NUM_SELECT_PINS; j++) {
                digitalWrite(PINS_SENSOR_SELECT[j], (i & (1 << j)) >> j);
            }
            Serial.print(analogRead(PIN_SENSOR_INPUT));
            Serial.print(",");
        }
    }
    if (millis() % ms_per_training_packet > 20) {
        must_deliminate_packet = 1;
    }
    // Newline
    Serial.println();
}
