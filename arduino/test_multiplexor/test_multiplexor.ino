int numSensors = 15;
int data[13];
int PIN_SENSOR_SELECT_0 = 8;
int PIN_SENSOR_SELECT_1 = 9;
int PIN_SENSOR_SELECT_2 = 10;
int PIN_SENSOR_SELECT_3 = 11;
int PIN_SENSOR_INPUT = A0;

void setup() {
    // Use a high baud rate so that the sensor readings are more accurate.
    Serial.begin(57600);
    pinMode(PIN_SENSOR_INPUT, INPUT);
    pinMode(PIN_SENSOR_SELECT_0, OUTPUT);
    pinMode(PIN_SENSOR_SELECT_1, OUTPUT);
    pinMode(PIN_SENSOR_SELECT_2, OUTPUT);
    pinMode(PIN_SENSOR_SELECT_3, OUTPUT);
}

void loop() {
    // Read each sensor value from the multiplexor.
    for (int i = 0; i < 16; i++) {
        digitalWrite(8, (i & 0b1000) >> 3);
        digitalWrite(9, (i & 0b0100) >> 2);
        digitalWrite(10, (i & 0b0010) >> 1);
        digitalWrite(11, (i & 0b0001) >> 0);
        Serial.print(analogRead(A0));
        Serial.print(",");
    }
    delay(5);
    Serial.println();
}
