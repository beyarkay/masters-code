/**
 * The (simplified) layout for the pins looks like:
 * 
 * pin A2<------+   +------>pin D7
 *              |   |
 * pin A3<---01 02 03 04--->pin D6
 *          switch numbers
 * pin A4<---01 02 03 04--->pin D5
 *              |   |
 * pin A5<------+   +------>pin D4
 * 
 * So the mapping of pins to switch numbers for the Top and Bottom
 * switches is:
 * - Top1 -> A3, Top2 -> A2, Top3 -> D7, Top4 -> D6
 * - Bot1 -> A4, Bot2 -> A5, Bot3 -> D4, Bot4 -> D5
 */
const int switches[] = {
     A3, A2, 7, 6, 
     A4, A5, 4, 5
};
const int pinsLen = 8;
const int buzzerPin = 2;
int selection = 200;


void setup() {
    for (int i = 0; i < pinsLen; i++) {
        pinMode(switches[i], INPUT_PULLUP);
    }
    // Buzzer
    pinMode(buzzerPin, OUTPUT);

    // Start up the serial writer
    Serial.begin(9600);
}

void loop() {
    selection = 0;
    for (int i = 0; i < pinsLen; i++) {
        int val = 1 - digitalRead(switches[i]);
        selection |= val << (pinsLen - i - 1);
        Serial.print(val);
        Serial.print(i == 3 ? " " : "");
    }
    Serial.print(" => ");
    Serial.println(selection);
    //beep(2);
    delay(1000);
}

void beep(int times) {
    beep(buzzerPin, times, 150);
}

void beep(int pin, int times, int duty) {
    analogWrite(pin, 0);
    for (int i = 0; i < times; i++) {
        analogWrite(pin, duty);
        delay(50);
        analogWrite(pin, 0);
        delay(100);
    }
}
