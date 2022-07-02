/*
   BLE_Peripheral.ino

   This program uses the ArduinoBLE library to set-up an Arduino Nano 33 BLE 
   as a peripheral device and specifies a service and a characteristic. Depending 
   of the value of the specified characteristic, an on-board LED gets on. 

   The circuit:
   - Arduino Nano 33 BLE. 

   This example code is in the public domain.
 */

#include <ArduinoBLE.h>

enum {
    GESTURE_NONE  = -1,
    GESTURE_UP    = 0,
    GESTURE_DOWN  = 1,
    GESTURE_LEFT  = 2,
    GESTURE_RIGHT = 3
};

const char* deviceServiceUuid               = "19b10000-e8f2-537e-4f6c-d104768a1214";
const char* deviceServiceCharacteristicUuid = "19b10001-e8f2-537e-4f6c-d104768a1214";

int gesture = -1;

BLEService gest_service(deviceServiceUuid); 
BLEByteCharacteristic gest_charac(deviceServiceCharacteristicUuid, BLERead | BLEWrite);


void setup() {
    Serial.begin(9600);
    while (!Serial);  

    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    if (!BLE.begin()) {
        Serial.println("- Starting Bluetooth¨ Low Energy module failed!");
        while (1);
    }

    BLE.setLocalName("Arduino Nano 33 BLE (Peripheral)");
    // Advertise the gesture service
    BLE.setAdvertisedService(gest_service);
    // Add the single characteristic to the gesture service
    gest_service.addCharacteristic(gest_charac);
    // Actually add the service to the list of services
    BLE.addService(gest_service);
    // Write a value of -1 to the service
    gest_charac.writeValue(-1);
    // Actually advertise
    BLE.advertise();

    Serial.println("Nano 33 BLE (Peripheral Device)");
}

void loop() {
    BLEDevice central = BLE.central();
    Serial.print(".");
    delay(500);

    if (central) {
        Serial.print("* Device MAC address: ");
        Serial.println(central.address());

        while (central.connected()) {
            gest_charac.writeValue(++gesture % 4);
            Serial.print("Wrote ");
            Serial.println(gesture % 4);
        }
        Serial.println("* Disconnected to central device!");
    }
}
