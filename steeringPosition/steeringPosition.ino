int a;
void setup() {
  // put your setup code here, to run once:
  pinMode(A1,INPUT);
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  a=analogRead(A1);
  a=1024-a;
  Serial.println(a);
  delay(10);
}
