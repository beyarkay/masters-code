#include <Wire.h>
const int NUM_SENSORS = 15;
const int NUM_SELECT_PINS = 4;
const int PINS_SENSOR_SELECT[] = {8, 9, 10, 11};
const int PIN_SENSOR_INPUT = A0;
int val = 0;
char newline[2] = "\n";

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    // register sensor selection pins as outputs
    for (int i = 0; i < NUM_SELECT_PINS; i++) {
        pinMode(PINS_SENSOR_SELECT[i], OUTPUT);
    }
    // register the sensor input pin as input
    pinMode(PIN_SENSOR_INPUT, INPUT);
    // Start up the I2C bus to communicate with the right hand
    Wire.begin();
}

void loop() {
    Wire.beginTransmission(4); // transmit to device #4
    digitalWrite(LED_BUILTIN, HIGH);
    for (int i = 0; i < NUM_SENSORS; i++) {
        for (int j = 0; j < NUM_SELECT_PINS; j++) {
            digitalWrite(PINS_SENSOR_SELECT[j], (i & (1 << j)) >> j);
        }
        val = analogRead(PIN_SENSOR_INPUT);
        char buffer[5];
        sprintf(buffer, "%4i,", val);
        Wire.write(buffer, 5);
    }
    Wire.write(newline, 2);
    delay(1);
    digitalWrite(LED_BUILTIN, LOW);
    Wire.endTransmission();
}

