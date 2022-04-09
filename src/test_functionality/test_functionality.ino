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
const int PINS_DIP_SWITCHES[] = {
     A3, A2, 7, 6, 
     A4, A5, 4, 5
};
const int num_dip_switches = 8;
const int PIN_BUZZER = 2;
int gesture_index = 200;

void setup() {
    for (int i = 0; i < num_dip_switches; i++) {
        pinMode(PINS_DIP_SWITCHES[i], INPUT_PULLUP);
    }
    // Buzzer
    pinMode(PIN_BUZZER, OUTPUT);

    // Start up the serial writer
    Serial.begin(9600);
}

void loop() {
    gesture_index = 0;
    for (int i = 0; i < num_dip_switches; i++) {
        int val = 1 - digitalRead(PINS_DIP_SWITCHES[i]);
        gesture_index |= val << (num_dip_switches - i - 1);
        Serial.print(val);
        Serial.print(i == 3 ? " " : "");
    }
    Serial.print(" => ");
    Serial.println(gesture_index);
    //beep(2);
    delay(1000);
}

void beep(int times) {
    beep(PIN_BUZZER, times, 150);
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
