const int DEBUG = 0;
const int NUM_SENSORS = 15;
const int PIN_SENSOR_SELECT_0 = 8;
const int PIN_SENSOR_SELECT_1 = 9;
const int PIN_SENSOR_SELECT_2 = 10;
const int PIN_SENSOR_SELECT_3 = 11;
const int PIN_SENSOR_INPUT = A0;
const int MS_PER_ANALOG_READ = 1;
// There are 8 DIP switches that are used to select a gesture index
const int PINS_DIP_SWITCHES[] = {
     A3, A2, 7, 6,
     A4, A5, 4, 5
};
const int num_dip_switches = 8;
// There are 255 indexable gestures 0x01 to 0xFF, with 0x00 reserved to
// indicate that the glove is in KEYBOARD mode and not in TRAINING mode.
int gesture_index = 0x01;
// The piezo-electric buzzer is used to indicate the boundary between training
// packets
const int PIN_BUZZER = 2;

void setup() {
    // Use a high baud rate so that the sensor readings are more accurate.
    Serial.begin(57600);
    Serial.println("low,high,gesture_idx,s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14");
    pinMode(PIN_SENSOR_SELECT_0, OUTPUT);
    pinMode(PIN_SENSOR_SELECT_1, OUTPUT);
    pinMode(PIN_SENSOR_SELECT_2, OUTPUT);
    pinMode(PIN_SENSOR_SELECT_3, OUTPUT);
    pinMode(PIN_SENSOR_INPUT, INPUT);
    for (int i = 0; i < num_dip_switches; i++) {
        pinMode(PINS_DIP_SWITCHES[i], INPUT_PULLUP);
    }
    // Buzzer
    pinMode(PIN_BUZZER, OUTPUT);
}

void loop() {
    // Print low and high so the y-axis doesn't auto-adjust
    Serial.print("0,1024,");

    // Read in the DIP switches to figure out what gesture index we're at
    // Init `gesture_index` to zero so bit shifts can work properly
    gesture_index = 0x00;
    for (int i = 0; i < num_dip_switches; i++) {
        int val = 1 - digitalRead(PINS_DIP_SWITCHES[i]);
        gesture_index |= val << (num_dip_switches - i - 1);
    }
    // Print the gesture_index, shifted so that it shows up on a [0,1024] scale
    Serial.print(gesture_index << 2);
    Serial.print(",");

    if (gesture_index == 0x00 &&
        millis() % 1000 <= MS_PER_ANALOG_READ * NUM_SENSORS) {
        analogWrite(PIN_BUZZER, 150);
        for (int i = 0; i < NUM_SENSORS; i++) {
            Serial.print("0,");
        }
    } else {
        if (millis() % 1000 <= 50)  {
            analogWrite(PIN_BUZZER, 150);
        } else {
            analogWrite(PIN_BUZZER, 0);
        }
        // Print each sensor value from the multiplexor
        for (int i = 0; i < NUM_SENSORS; i++) {
            digitalWrite(PIN_SENSOR_SELECT_0, (i & 0b0001) >> 0);
            digitalWrite(PIN_SENSOR_SELECT_1, (i & 0b0010) >> 1);
            digitalWrite(PIN_SENSOR_SELECT_2, (i & 0b0100) >> 2);
            digitalWrite(PIN_SENSOR_SELECT_3, (i & 0b1000) >> 3);
            // TODO does the multiplexor change fast enough that we dont' need
            // to wait for it with this delay?
            delay(MS_PER_ANALOG_READ);
            Serial.print(analogRead(PIN_SENSOR_INPUT));
            Serial.print(",");
        }
    }
    // Newline
    Serial.println();
}
