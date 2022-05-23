#include <Wire.h>

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);
    Wire.begin(4);
    Wire.onReceive(receiveEvent);
    Serial.begin(57600);
}

// `loop()` is required for the .ino file to compile
void loop() {
    delay(1);
}

void receiveEvent(int num_events) {
    digitalWrite(LED_BUILTIN, HIGH);
    while (Wire.available() > 0) {
        char c = Wire.read();
        // if (c == '\n') {
        //     Serial.println("");
        // } else {
        Serial.print(c);
        // }
    }
    delay(5);
    digitalWrite(LED_BUILTIN, LOW);
}

