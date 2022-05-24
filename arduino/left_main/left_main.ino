#include <Wire.h>
const int NUM_SENSORS = 15;
const int NUM_SELECT_PINS = 4;
const int PINS_SENSOR_SELECT[] = {8, 9, 10, 11};
const int PIN_SENSOR_INPUT = A0;
int val = 0;

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
        int thou = val / 1000;
        if (thou > 0) {
            Wire.write('0' + thou);
            val -= thou * 1000;
        }
        int hund = val / 100;
        if (hund > 0 || thou) {
            Wire.write('0' + hund);
            val -= hund * 100;
        }
        int tens = val / 10;
        if (tens > 0 || hund || thou) {
            Wire.write('0' + tens);
            val -= tens * 10;
        }
        if (val > 0 || tens || hund || thou) {
            Wire.write('0' + val);
        }
        Wire.write(',');
    }
    // Wire.write('!');
    delay(5);
    digitalWrite(LED_BUILTIN, LOW);
    int status = Wire.endTransmission();
    /* Status codes: 
        0. Success.
        1. Data too long to fit in transmit buffer.
        2. Received NACK on transmit of address.
        3. Received NACK on transmit of data.
        4. Other error.
        5. Timeout
    */
    for (int i = 0; i < status; i++) {
        digitalWrite(LED_BUILTIN, HIGH);
        delay(500);
        digitalWrite(LED_BUILTIN, LOW);
        delay(500);
    }
}

