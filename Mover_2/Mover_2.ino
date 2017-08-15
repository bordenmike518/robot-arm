
#include <ax12.h>
#include <BioloidController.h>
#include "poses.h"



BioloidController bioloid = BioloidController(1000000);

const int SERVOCOUNT = 6;
int id;
int pos;


void setup()
{
  id = 1;
  pos = 0;    
  
  Serial.begin(9600);
  delay(500);
  Serial.println("*********************************");
  Serial.println("Serial Communication Established.");
  Serial.println("*********************************");
  Serial.println("");
  Serial.println("");
  Serial.println("");
  
  MoveToZero();
  
  delay(5000);
  
  Relax(1);
  Relax(2);
  Relax(3);
  Relax(4);
  Relax(5);
  Relax(6);
    
  delay(100);//wait for servo to move
}

void loop()
{
    
  Serial.println("********************************");
  Serial.println("********************************");  
  
  
    
  Serial.print("Servo 1 = ");
  Serial.println(GetPosition(1));
    
  Serial.print("Servo 2 = ");
  Serial.println(GetPosition(2));
    
  Serial.print("Servo 3 = ");
  Serial.println(GetPosition(3));
    
  Serial.print("Servo 4 = ");
  Serial.println(GetPosition(4));
    
  Serial.print("Servo 5 = ");
  Serial.println(GetPosition(5));
    
  Serial.print("Servo 6 = ");
  Serial.println(GetPosition(6));
  delay(3000);
}

void MoveToZero()
{
  delay(100);
  bioloid.loadPose(Home);
  bioloid.readPose();
  Serial.println("##################################");
  Serial.println("Moving servos to centered position");
  Serial.println("##################################");
  delay(1000);
  bioloid.interpolateSetup(1000);
  while (bioloid.interpolating > 0)
  {
    bioloid.interpolateStep();
    delay(0.5);
  }
}

void RelaxAll()
{
  Relax(1);
  Relax(2);
  Relax(3);
  Relax(4);
  Relax(5);
  Relax(6);
}

