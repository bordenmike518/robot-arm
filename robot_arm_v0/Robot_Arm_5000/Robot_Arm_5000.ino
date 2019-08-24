// import ax12 library to send DYNAMIXEL commands
#include <ax12.h>

// Constructor
void setup()
{
    Serial.begin(9600);
    int ZERO_POS[6] = {512, 512, 512, 512, 512, 512};
    int HOME_POS[6] = {512, 650, 512, 212, 512, 670};
    int LOOK_RIGHT[6] = {511, 651, 464, 124, 882, 388};
    int LOOK_LEFT[6] = {511, 634, 584, 133, 239, 389};

    moveServoArray(ZERO_POS, 40);
    delay(4600);
    
    moveServoArray(HOME_POS, 30);
    delay(5000);
    
    moveServoArray(LOOK_LEFT, 45);
    delay(3000);
    
    moveServoArray(LOOK_RIGHT, 45);
   
}

// While running...
void loop()
{
  
}

void moveServo(int id, int pos, int spd) {
    ax12SetRegister2(id, AX_GOAL_SPEED_L, spd);
    ax12SetRegister2(id, AX_GOAL_POSITION_L, pos);
}

void moveServoArray(int arr[][], int spd) {
    int i, j;
    for (i = 0; i < 200; i++) {
      for(j = 0; j < 6; j++){
        moveServo(j + 1, arr[i][j], spd);
      }
    }
}

void get_position() {
    int i;
    char str2Print;
    for (i = 0; i < 6; i++) {
        Relax(i + 1);
    }
    delay(5000);
    for (i = 0; i < 6; i++) {
        Serial.println(GetPosition(i + 1));
    }  
}



