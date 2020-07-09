#include <Servo.h>
Servo myservo1;
Servo myservo2;// create servo object to control a servo
// twelve servo objects can be created on most boards
Servo myservo3;
Servo myservo4;
int pos = 0, a1,b1;    // variable to store the servo positionc;
String c;
String a,b;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  myservo1.attach(9);
  myservo3.attach(2);
   myservo4.attach(5);
  myservo2.attach(10);
  pinMode(5,OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available()>0)
  {
    c=Serial.read();
    int x = c.indexOf(',');
    a = c.substring(0,x);
    b= c.substring(x+1,(c.length()));
   //Serial.println(a);
    //Serial.println(b);
    a1=atoi(a.c_str())*20 + 150;
    myservo1.write(a1);
    myservo4.write(a1);
    //Serial.println(a);
     //delay(500); 
    b1=atoi(b.c_str())*20 + 150;
    myservo3.write(b1);
    myservo2.write(b1);
    //analogWrite(5,b1);
    delay(200);
    
         // waits 15ms for the servo to reach the position
  }
  }
