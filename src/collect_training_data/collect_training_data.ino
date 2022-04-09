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
    Serial.println("low,high,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16");
    pinMode(PIN_SENSOR_INPUT, INPUT);
    pinMode(PIN_SENSOR_SELECT_0, OUTPUT);
    pinMode(PIN_SENSOR_SELECT_1, OUTPUT);
    pinMode(PIN_SENSOR_SELECT_2, OUTPUT);
    pinMode(PIN_SENSOR_SELECT_3, OUTPUT);
}

void loop() {
    
    // Read each sensor value from the multiplexor.
    for (int i = 0; i < 16; i++) {
        digitalWrite(PIN_SENSOR_SELECT_0, (i & 0b1000) >> 3);
        digitalWrite(PIN_SENSOR_SELECT_1, (i & 0b0100) >> 2);
        digitalWrite(PIN_SENSOR_SELECT_2, (i & 0b0010) >> 1);
        digitalWrite(PIN_SENSOR_SELECT_3, (i & 0b0001) >> 0);
        Serial.print(analogRead(PIN_SENSOR_INPUT));
        Serial.print(",");
    }
    delay(5);
    Serial.println();
}
