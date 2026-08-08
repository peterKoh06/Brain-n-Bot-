#include <Arduino.h>

void setup() {
  pinMode(11, OUTPUT);          // Onboard LED is on pin 11 (Arduino numbering)
  Serial.begin(9600);           // USB Serial
  while (!Serial && millis() < 3000);  // Wait up to 3 s for Serial Monitor
  Serial.println("Teensy 2.0 ready");
}

void loop() {
  digitalWrite(11, HIGH);
  Serial.println("LED ON");
  delay(500);
  digitalWrite(11, LOW);
  Serial.println("LED OFF");
  delay(500);
}