/*
Team Id: HC#3266
Author List: Hemanth Kumar K.L , Aman Bhat T, Mahantesh R, Nischith B.O.
File Name: White_Line_Following.c
Theme: Homecoming(Task-4, Original configuration)
Functions (important): Nodes, Path, Anipick, Habidrop, Pick, Drop, Facing, Straight, Turn, Semiturn, Nodetraverse, Turnsoft
Global Variables: data,serial_ip,processed_ip,z,count, f,b,r,l,a,p,mapping
*/



#include<avr/io.h>
#include<avr/interrupt.h>
#include<util/delay.h>
#include "lcd.c"

#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<string.h>

int* nodes(int,int,int);
void path(int,int,int,int);
int turn(int);
void straight(int);
int turn_180();
void facing(int);
void semiturn(int);
void habidrop(int ,int ,int );
void anipick(char* );
void nodetraverse(int);
void port_init();
void port_init1();
void timer5_init();
void velocity(unsigned char, unsigned char);
void motors_delay();
int turnsoft(int);
void pick();
void drop();
void init_devices();
void init_devices1();

unsigned char ADC_Conversion(unsigned char);
unsigned char ADC_Value;
unsigned char flag = 0;
unsigned char Left_white_line = 0;
unsigned char Center_white_line = 0;
unsigned char Right_white_line = 0;

unsigned long int ShaftCountLeft = 0; //to keep track of left position encoder
unsigned long int ShaftCountRight = 0; //to keep track of right position encoder
unsigned int Degrees; //to accept angle in degrees for turning

unsigned char data; //to store received data from UDR2
char serial_ip[100],processed_ip[100];
int z=0,count=0;
int f=1,b=0,r=0,l=0,a=4,p=0,q=0;
char mapping[30][30];

void servo1_pin_config (void)
{
	DDRB  = DDRB | 0x20;  //making PORTB 5 pin output
	PORTB = PORTB | 0x20; //setting PORTB 5 pin to logic 1
}

//Configure PORTB 6 pin for servo motor 2 operation
void servo2_pin_config (void)
{
	DDRB  = DDRB | 0x40;  //making PORTB 6 pin output
	PORTB = PORTB | 0x40; //setting PORTB 6 pin to logic 1
}

void buzzer_pin_config (void)
{
 DDRC = DDRC | 0x08;		//Setting PORTC 3 as outpt
 PORTC = PORTC & 0xF7;		//Setting PORTC 3 logic low to turnoff buzzer
}

//Function to initialize ports
void interrupt_switch_config (void)
{
	DDRE = DDRE & 0x7F;  //PORTE 7 pin set as input
	PORTE = PORTE | 0x80; //PORTE7 internal pull-up enabled
}


void buzzer_on (void)
{
 unsigned char port_restore = 0;
 port_restore = PINC;
 port_restore = port_restore | 0x08;
 PORTC = port_restore;
}


void buzzer_off (void)
{
	unsigned char port_restore = 0;
	port_restore = PINC;
	port_restore = port_restore & 0xF7;
	PORTC = port_restore;
}


//Function To Initialize UART2
// desired baud rate:9600
// actual baud rate:9600 (error 0.0%)
// char size: 8 bit
// parity: Disabled
void uart2_init(void)
{
 UCSR2B = 0x00; //disable while setting baud rate
 UCSR2A = 0x00;
 UCSR2C = 0x06;
 UBRR2L = 0x5F; //set baud rate lo
 UBRR2H = 0x00; //set baud rate hi
 UCSR2B = 0x98;
}


SIGNAL(SIG_USART2_RECV) 		// ISR for receive complete interrupt
{
	data = UDR2; 				//making copy of data from UDR2 in 'data' variable

	UDR2 = data; 				//echo data back to PC

	if(data)
		serial_ip[z++]=data;
}






//Function to configure ports to enable robot's motion
void motion_pin_config (void)
{
 DDRA = DDRA | 0x0F;
 PORTA = PORTA & 0xF0;
 DDRL = DDRL | 0x18;   //Setting PL3 and PL4 pins as output for PWM generation
 PORTL = PORTL | 0x18; //PL3 and PL4 pins are for velocity control using PWM.
}

//Function to configure INT4 (PORTE 4) pin as input for the left position encoder
void left_encoder_pin_config (void)
{
 DDRE  = DDRE & 0xEF;  //Set the direction of the PORTE 4 pin as input
 PORTE = PORTE | 0x10; //Enable internal pull-up for PORTE 4 pin
}

//Function to configure INT5 (PORTE 5) pin as input for the right position encoder
void right_encoder_pin_config (void)
{
 DDRE  = DDRE & 0xDF;  //Set the direction of the PORTE 4 pin as input
 PORTE = PORTE | 0x20; //Enable internal pull-up for PORTE 4 pin
}

void port_init1(void)
{
	servo1_pin_config(); //Configure PORTB 5 pin for servo motor 1 operation
	servo2_pin_config(); //Configure PORTB 6 pin for servo motor 2 operation
}

//Function to initialize ports
void port_init()
{
 motion_pin_config(); //robot motion pins config
 left_encoder_pin_config(); //left encoder pin config
 right_encoder_pin_config(); //right encoder pin config
 lcd_port_config();
 adc_pin_config();
 buzzer_pin_config();
 interrupt_switch_config();
}

void timer1_init(void)
{
 TCCR1B = 0x00; //stop
 TCNT1H = 0xFC; //Counter high value to which OCR1xH value is to be compared with
 TCNT1L = 0x01;	//Counter low value to which OCR1xH value is to be compared with
 OCR1AH = 0x03;	//Output compare Register high value for servo 1
 OCR1AL = 0xFF;	//Output Compare Register low Value For servo 1
 OCR1BH = 0x03;	//Output compare Register high value for servo 2
 OCR1BL = 0xFF;	//Output Compare Register low Value For servo 2
 OCR1CH = 0x03;	//Output compare Register high value for servo 3
 OCR1CL = 0xFF;	//Output Compare Register low Value For servo 3
 ICR1H  = 0x03;
 ICR1L  = 0xFF;
 TCCR1A = 0xAB; /*{COM1A1=1, COM1A0=0; COM1B1=1, COM1B0=0; COM1C1=1 COM1C0=0}
 					For Overriding normal port functionality to OCRnA outputs.
				  {WGM11=1, WGM10=1} Along With WGM12 in TCCR1B for Selecting FAST PWM Mode*/
 TCCR1C = 0x00;
 TCCR1B = 0x0C; //WGM12=1; CS12=1, CS11=0, CS10=0 (Prescaler=256)
}

void left_position_encoder_interrupt_init (void) //Interrupt 4 enable
{
 cli(); //Clears the global interrupt
 EICRB = EICRB | 0x02; // INT4 is set to trigger with falling edge
 EIMSK = EIMSK | 0x10; // Enable Interrupt INT4 for left position encoder
 sei();   // Enables the global interrupt
}

void right_position_encoder_interrupt_init (void) //Interrupt 5 enable
{
 cli(); //Clears the global interrupt
 EICRB = EICRB | 0x08; // INT5 is set to trigger with falling edge
 EIMSK = EIMSK | 0x20; // Enable Interrupt INT5 for right position encoder
 sei();   // Enables the global interrupt
}

//ISR for right position encoder
ISR(INT5_vect)
{
 ShaftCountRight++;  //increment right shaft position count
}


//ISR for left position encoder
ISR(INT4_vect)
{
 ShaftCountLeft++;  //increment left shaft position count
}


//Function used for setting motor's direction
void motion_set (unsigned char Direction)
{
 unsigned char PortARestore = 0;

 Direction &= 0x0F; 		// removing upper nibbel for the protection
 PortARestore = PORTA; 		// reading the PORTA original status
 PortARestore &= 0xF0; 		// making lower direction nibbel to 0
 PortARestore |= Direction; // adding lower nibbel for forward command and restoring the PORTA status
 PORTA = PortARestore; 		// executing the command
}

void forward (void) //both wheels forward
{
  motion_set(0x06);
}

void back (void) //both wheels backward
{
  motion_set(0x09);
}

void left (void) //Left wheel backward, Right wheel forward
{
  motion_set(0x05);
}

void right (void) //Left wheel forward, Right wheel backward
{
  motion_set(0x0A);
}

void soft_left (void) //Left wheel stationary, Right wheel forward
{
 motion_set(0x04);
}

void soft_right (void) //Left wheel forward, Right wheel is stationary
{
 motion_set(0x02);
}

void soft_left_2 (void) //Left wheel backward, right wheel stationary
{
 motion_set(0x01);
}

void soft_right_2 (void) //Left wheel stationary, Right wheel backward
{
 motion_set(0x08);
}

void stop (void)
{
  motion_set(0x00);
}


//Function to configure LCD port
void lcd_port_config (void)
{
 DDRC = DDRC | 0xF7; //all the LCD pin's direction set as output
 PORTC = PORTC & 0x80; // all the LCD pins are set to logic 0 except PORTC 7
}

//ADC pin configuration
void adc_pin_config (void)
{
 DDRF = 0x00;
 PORTF = 0x00;
 DDRK = 0x00;
 PORTK = 0x00;
}



// Timer 5 initialized in PWM mode for velocity control
// Prescale:256
// PWM 8bit fast, TOP=0x00FF
// Timer Frequency:225.000Hz
void timer5_init()
{
	TCCR5B = 0x00;	//Stop
	TCNT5H = 0xFF;	//Counter higher 8-bit value to which OCR5xH value is compared with
	TCNT5L = 0x01;	//Counter lower 8-bit value to which OCR5xH value is compared with
	OCR5AH = 0x00;	//Output compare register high value for Left Motor
	OCR5AL = 0xFF;	//Output compare register low value for Left Motor
	OCR5BH = 0x00;	//Output compare register high value for Right Motor
	OCR5BL = 0xFF;	//Output compare register low value for Right Motor
	OCR5CH = 0x00;	//Output compare register high value for Motor C1
	OCR5CL = 0xFF;	//Output compare register low value for Motor C1
	TCCR5A = 0xA9;	/*{COM5A1=1, COM5A0=0; COM5B1=1, COM5B0=0; COM5C1=1 COM5C0=0}
 					  For Overriding normal port functionality to OCRnA outputs.
				  	  {WGM51=0, WGM50=1} Along With WGM52 in TCCR5B for Selecting FAST PWM 8-bit Mode*/

	TCCR5B = 0x0B;	//WGM12=1; CS12=0, CS11=1, CS10=1 (Prescaler=64)
}

void adc_init()
{
	ADCSRA = 0x00;
	ADCSRB = 0x00;		//MUX5 = 0
	ADMUX = 0x20;		//Vref=5V external --- ADLAR=1 --- MUX4:0 = 0000
	ACSR = 0x80;
	ADCSRA = 0x86;		//ADEN=1 --- ADIE=1 --- ADPS2:0 = 1 1 0
}

//Function For ADC Conversion
unsigned char ADC_Conversion(unsigned char Ch)
{
	unsigned char a;
	if(Ch>7)
	{
		ADCSRB = 0x08;
	}
	Ch = Ch & 0x07;
	ADMUX= 0x20| Ch;
	ADCSRA = ADCSRA | 0x40;		//Set start conversion bit
	while((ADCSRA&0x10)==0);	//Wait for conversion to complete
	a=ADCH;
	ADCSRA = ADCSRA|0x10; //clear ADIF (ADC Interrupt Flag) by writing 1 to it
	ADCSRB = 0x00;
	return a;
}

//Function To Print Sesor Values At Desired Row And Coloumn Location on LCD
void print_sensor(char row, char coloumn,unsigned char channel)
{

	ADC_Value = ADC_Conversion(channel);
	lcd_print(row, coloumn, ADC_Value, 3);
}





//Function used for turning robot by specified degrees
void angle_rotate(unsigned int Degrees)
{
 float ReqdShaftCount = 0;
 unsigned long int ReqdShaftCountInt = 0;

 ReqdShaftCount = (float) Degrees/ 4.090; // division by resolution to get shaft count
 ReqdShaftCountInt = (unsigned int) ReqdShaftCount;
 ShaftCountRight = 0;
 ShaftCountLeft = 0;

 while (1)
 {
  if((ShaftCountRight >= ReqdShaftCountInt) | (ShaftCountLeft >= ReqdShaftCountInt))
  break;
 }
 stop(); //Stop robot
}

//Function used for moving robot forward by specified distance

void linear_distance_mm(unsigned int DistanceInMM)
{
 float ReqdShaftCount = 0;
 unsigned long int ReqdShaftCountInt = 0;

 ReqdShaftCount = DistanceInMM / 5.338; // division by resolution to get shaft count
 ReqdShaftCountInt = (unsigned long int) ReqdShaftCount;

 ShaftCountRight = 0;
 while(1)
 {
  if(ShaftCountRight > ReqdShaftCountInt)
  {
  	break;
  }
 }
 stop(); //Stop robot
}


//Function for velocity control
void velocity (unsigned char left_motor, unsigned char right_motor)
{
	OCR5AL = (unsigned char)left_motor;
	OCR5BL = (unsigned char)right_motor;
}



void forward_mm(unsigned int DistanceInMM)
{
 forward();
 linear_distance_mm(DistanceInMM);
}

void back_mm(unsigned int DistanceInMM)
{
 back();
 linear_distance_mm(DistanceInMM);
}

void left_degrees(unsigned int Degrees)
{
// 88 pulses for 360 degrees rotation 4.090 degrees per count
 left(); //Turn left

 angle_rotate(Degrees);
}



void right_degrees(unsigned int Degrees)
{
// 88 pulses for 360 degrees rotation 4.090 degrees per count
 right(); //Turn right
 angle_rotate(Degrees);
}


void soft_left_degrees(unsigned int Degrees)
{
 // 176 pulses for 360 degrees rotation 2.045 degrees per count
 soft_left(); //Turn soft left
 Degrees=Degrees*2;
 angle_rotate(Degrees);
}

void soft_right_degrees(unsigned int Degrees)
{
 // 176 pulses for 360 degrees rotation 2.045 degrees per count
 soft_right();  //Turn soft right
 Degrees=Degrees*2;
 angle_rotate(Degrees);
}

void soft_left_2_degrees(unsigned int Degrees)
{
 // 176 pulses for 360 degrees rotation 2.045 degrees per count
 soft_left_2(); //Turn reverse soft left
 Degrees=Degrees*2;
 angle_rotate(Degrees);
}

void soft_right_2_degrees(unsigned int Degrees)
{
 // 176 pulses for 360 degrees rotation 2.045 degrees per count
 soft_right_2();  //Turn reverse soft right
 Degrees=Degrees*2;
 angle_rotate(Degrees);
}

void init_devices1(void)
{
	cli(); //disable all interrupts
	port_init1();
	timer1_init();
	sei(); //re-enable interrupts
}

void init_devices (void)
{
 	cli(); //Clears the global interrupts
	port_init();  //Initializes all the ports
	left_position_encoder_interrupt_init();
 	right_position_encoder_interrupt_init();
	adc_init();
	timer5_init();
	uart2_init(); //Initailize UART1 for serial communiaction
	port_init();
	//timer1_init();
	sei();   //Enables the global interrupts
}

//Function to rotate Servo 1 by a specified angle in the multiples of 0.4 degrees
void servo_1(unsigned char degrees)
{
	float PositionPanServo = 0;
	PositionPanServo = ((float)degrees / 0.4) + 35.0;     //start from an offset of 35
	OCR1AH = 0x00;
	OCR1AL = (unsigned char) PositionPanServo;
}



//Function to rotate Servo 2 by a specified angle in the multiples of 0.9 degrees
void servo_2(unsigned char degrees)
{
	float PositionTiltServo = 0;
	PositionTiltServo = ((float)degrees / 0.9)+40.0;   //start from an offset of 40
	OCR1BH = 0x00;
	OCR1BL = (unsigned char) PositionTiltServo;
}

void servo_1_free (void) //makes servo 1 free rotating
{
	OCR1AH = 0x03;
	OCR1AL = 0xFF; //Servo 1 off
}

void servo_2_free (void) //makes servo 2 free rotating
{
	OCR1BH = 0x03;
	OCR1BL = 0xFF; //Servo 2 off
}

/* Finds all nodes of a given habitat and returns the coordinates of the nearest node from animal location i.e. (x,y) */
int* nodes(int n,int x,int y)
{
	/*
	Function name: nodes
	Input: n,x,y
	Output: temp
	*/

	// find coordinates of bottom left corner of habitat number n
	// ----------------------------------------------------------
	int x0=n%5;
	int y0=n/5;

	// Calculate coordinates of other 3 nodes of habitat n and returns the closest to animal location
	// ----------------------------------------------------------------------------------------------
	int x1,y1,x2,y2,x3,y3,rx,ry,d1,d2;
	int *temp=(int*) malloc(sizeof(int)*2);
	x1=x0+1;
	x2=x1;
	x3=x0;
	y1=y0;
	y2=y0+1;
	y3=y2;
	d1=abs(x-x0);
	d2=abs(x-x1);

	d1>d2?((abs(y-y1)>abs(y-y2)?(rx=x1,ry=y2):(rx=x1,ry=y1))):((abs(y-y0)>abs(y-y3))?(rx=x0,ry=y3):(rx=x0,ry=y0)); // Find node with least distance

	// Store coordinates of closest node in temp and return it
	*temp=rx;
	*(temp+1)=ry;
	return temp;
}

/* Follow black line for given number of nodes(d) */
void straight(int d)
{
	nodetraverse(d);
}

/* Rotate 180 degrees from current direction on to black line */
int turn_180()
{
	/*
	Function Name: turn_180
	Input: None
	Output:a
	*/
	init_devices();
	right_degrees(160);
	while(1)
	{
		// Adjust position by rotating in steps of 3 degrees until black line is found
		Center_white_line = ADC_Conversion(2);
		if(Center_white_line<0x25)
			right_degrees(2);
		else
			break;
	}
	stop();
	_delay_ms(100);

	a=a+2;
	// Update the direction reference of the robot and return it
	if(a==8||a==0)
		a=4;
	return a;
}

/* Turn the robot by 90 degrees left/right based on parameter value*/
int turn(int f)
{
	/*
	Function Name: turn
	Input: f
	Output: a
	*/
	init_devices();
	forward_mm(12);
	stop();
	_delay_ms(100);

	// Rotate right by 90 degrees
	if(f==1)
	{
		soft_right_degrees(75);
		while(1)
		{
			// Adjust position by rotating in steps of 3 degrees until black line is found
			Center_white_line = ADC_Conversion(2);
			Left_white_line = ADC_Conversion(3);
			Right_white_line = ADC_Conversion(1);
			if(Center_white_line<0x20)
				soft_right_degrees(2);
			else
				break;
		}
		a++;
	}

	// Rotate left by 90 degrees
	else
	{
		soft_left_degrees(75);
		while(1)
		{
			// Adjust position by rotating in steps of 3 degrees until black line is found
			Center_white_line = ADC_Conversion(2);
			if(Center_white_line<0x20)
				soft_left_degrees(2);
			else
				break;
		}
		a--;
	}
	stop();
	_delay_ms(100);

	// Update direction reference of robot and return it
	if(a==8||a==0)
		a=4;
	return a;
}

/* Updates the direction flags based on direction reference value passed*/
void facing(int a)
{
	/*
	Function Name: facing
	Input: a
	Output: None
	*/
	if(a%4==0) // facing front
	{
		l=0;
		f=1;
		r=b=0;
	}
	else if(a%4==2) // facing back
	{
		l=f=r=0;
		b=1;
	}
	else if(a%4==1) // facing right
	{
		r=1;
		f=b=l=0;
	}
	else if(a%4==3) // facing left
	{
		l=1;
		f=b=r=0;
	}
	return;
}


/* Move the robot from source(x,y) to destination (rx,ry)*/
void path(int x,int y,int rx,int ry)
{
	/*
	Function Name: path
	Input: x,y,rx,ry
	Output: None
	*/
	if(b==1) // Swap the direction flags if robot is facing backward
	{
		p=r;r=l;l=p;
	}
	if(x==rx) // If destination has the same x-coordinate as source
	{
		if(ry>y) // If destination y-coordinate is greater than source
		{
			/* Turn the robot along y-direction */
			if(r==1 || l==1)
			{
				a=turn(l);
				facing(a);
			}
			else if(b==1)
			{
				a=turn_180();
				facing(a);
			}

			// Move straight for ry-y nodes
			straight(ry-y);
		}
		else if(ry<y) // If destination y-coordinate is lesser than source
		{
			/* Turn the robot along y-direction */
			if(r==1 || l==1)
			{
				a=turn(r);
				facing(a);
			}
			else if(f==1)
			{
				a=turn_180();
				facing(a);
			}

			// Move straight for y-ry nodes
			straight(y-ry);
		}
	}
	else if(rx>x) // If destination x-coordinate is greater than source
	{
		/* Turn the robot along x direction */
		if(r==0 && l==0)
		{
			a=turn(f);
			facing(a);
		}
		else if(l==1)
		{
			a=turn_180();
			facing(a);
		}

		// Move straight for rx-x nodes
		straight(rx-x);
		// Recursively call the path function
		path(rx,y,rx,ry);
	}
	else // If destination x-coordinate is lesser than source
	{
		/* Turn the robot along x direction */
		if(r==0 && l==0)
		{
			a=turn(b);
			facing(a);
		}
		else if(r==1)
		{
			a=turn_180();
			facing(a);
		}

		// Move straight for x-rx nodes
		straight(x-rx);
		// Recursively call the path function
		path(rx,y,rx,ry);
	}
	return;
}


/* Function to place the animals at their respective habitats */
void habidrop(int n,int x,int y)
{
	/*
	Function Name: Habidrop
	Input: n,x,y
	Output: None
	*/
	int x11=n%5;
	int y11=n/5;
	int node=0;
	if(x==x11)
	{
		if(y==y11) // Nearest node is bottom left corner of habitat
		{
			node=1;

			/* Turn the robot towards the habitat */
			if(f==1)
				semiturn(f);
			else if(r==1)
				semiturn(l);
			else
			{
				a=turn(l);
				facing(a);
				habidrop(n,x,y);
			}
		}
		else // Nearest node is top-left corner of habitat
		{
			node=4;

			/* Turn the robot towards the habitat */
			if(b==1)
				semiturn(f);
			else if(r==1)
				semiturn(r);
			else
			{
				a=turn(r);
				facing(a);
				habidrop(n,x,y);
			}
		}
	}
	else
	{
		if(y==y11) // Nearest node is at bottom-right corner of habitat
		{
			node=2;

			/* Turn the robot towards the habitat */
			if(l==1)
				semiturn(l);
			else if(f==1)
				semiturn(b);
			else
			{
				a=turn(b);
				facing(a);
				habidrop(n,x,y);
			}
		}
		else // Nearest node is at top-right corner of habitat
		{
			node=3;

			/* Turn the robot towards the habitat */
			if(b==1)
				semiturn(b);
			else if(l==1)
				semiturn(r);
			else
			{
				a=turn(b);
				facing(a);
				habidrop(n,x,y);
			}
		}
	}
	return;
}

/* Function to perform soft turns(keeping one wheel stationary and other wheel backward) */
int turnsoft(int f)
{
	/*
	Function Name: turnsoft
	Input: f
	Output: a
	*/
	init_devices();
	forward_mm(173);
	stop();
	_delay_ms(100);

	if(f==1) // Soft right
	{
		soft_right_2_degrees(105);
		while(1)
		{
			// Adjust position by rotating in steps of 3 degrees until black line is found
			Center_white_line = ADC_Conversion(2);
			if(Center_white_line<0x40)
				soft_right_2_degrees(2);
			else
				break;
		}
		a++;
	}
	else // Soft left
	{
		soft_left_2_degrees(80);
		while(1)
		{
			// Adjust position by rotating in steps of 3 degrees until black line is found
			Center_white_line = ADC_Conversion(2);
			if(Center_white_line<0x20)
				soft_left_2_degrees(2);
			else
				break;
		}
		a--;
	}
	stop();
	_delay_ms(100);

	// Update the direction reference and return it
	if(a==8||a==0)
		a=4;
	return a;
}

/* Function to pick the animals at given location (pos) */
void anipick(char* pos)
{
	/*
	Function Name: anipick
	Input: pos
	Output: None
	*/
	init_devices();
	if(pos[0]=='A') // If animal is at left side of arena
	{
		if(b==1 || f==1)
		{  
			if(pos[1]!='1')		// Normal cases
			{
			   a=turnsoft(b);
			   facing(a);
			   forward_mm(25);
			   stop();
			   _delay_ms(50);
			   pick();
			   back_mm(85);
			   stop();
			   _delay_ms(50);
			}			 
			/* Handling corner cases */
			else if(pos[1]=='6')
			{
				soft_right_2_degrees(15);
				forward_mm(50);
				pick();
				back_mm(35);
				stop();
				_delay_ms(100);
			}
			else if(pos[1]=='1')
			{
				forward_mm(30);
				stop();
				_delay_ms(50);
				soft_left_2_degrees(100);
				stop();
				_delay_ms(50);
				forward_mm(20);
				pick();
				back_mm(20);
				stop();
				_delay_ms(100);
				soft_right_degrees(94);
				stop();
				_delay_ms(50);
				stop();
				_delay_ms(50);
			}
		}

		// If robot is already at the animal's location
		else
		   { 
				forward_mm(50);
				stop();
				_delay_ms(50);
				pick();
				back_mm(55);
				stop();
				_delay_ms(50);
		   }			
	}

	else if(pos[0]=='F') // If animal is at right side of the arena
	{
		if(f==1 || b==1)
		{   
			if(pos[1]!='1')		//Normal cases
		    {   a=turnsoft(f);
			   facing(a);
		    }			
			
			/* Handling corner cases */
			if(pos[1]=='6')
			{
				soft_left_2_degrees(15);
				forward_mm(50);
				pick();
				back_mm(35);
				stop();
				_delay_ms(100);
			}
			else if(pos[1]=='1')	
			{   forward_mm(100);
				stop();
				_delay_ms(50);
				soft_left_2_degrees(40);
				stop();
				_delay_ms(50);
				forward_mm(30);
				pick();
				back_mm(30);
				stop();
				_delay_ms(100);
				soft_right_degrees(45);
				stop();
				_delay_ms(50);
				back_mm(125);
				stop();
				_delay_ms(50);
			}

			// Normal cases
			else
			{
				forward_mm(25);
				stop();
				_delay_ms(50);
				pick();
				back_mm(85);
				stop();
				_delay_ms(50);
			}
		}
		// If robot is already at the animal's location
		else
		{
			forward_mm(50);
			stop();
			_delay_ms(50);
			pick();
			back_mm(55);
			stop();
			_delay_ms(50);
		}
	}

	else if(pos[1]=='1' && pos[0]!='A' && pos[0]!='F') // If animal is at the bottom part of the arena
	{
		if(r==1 || l==1)
		{
			a=turnsoft(r);
			facing(a);
			forward_mm(25);
			stop();
			_delay_ms(50);
			pick();
			back_mm(85);
			stop();
			_delay_ms(50);
		}
		// If robot is already at the animal's location
		else
		{
			forward_mm(50);
			stop();
			_delay_ms(50);
			pick();
			back_mm(60);
			stop();
			_delay_ms(50);
		}	
	}
	else if(pos[1]=='6' && pos[0]!='A' && pos[0]!='F') // If animal is at top part of the arena
	{
		if(l==1 || r==1)
		{
			a=turnsoft(l);
			facing(a);
			forward_mm(25);
			stop();
			_delay_ms(50);
			pick();
			back_mm(85);
			stop();
			_delay_ms(50);
			
		}
		// If robot is already at the animal's location
		else
		{
			forward_mm(50);
			stop();
			_delay_ms(50);
			pick();
			back_mm(55);
			stop();
			_delay_ms(50);
		}	
	}
	return;
}

/* Function to turn the robot towards its habitat */
void semiturn(int s)
{
	/*
	Function Name: semiturn
	Input: s
	Output: None
	*/
	init_devices();
	if(s==1)
	{
		right_degrees(60);		
		//soft_right_degrees(deg);
	}	
	else
		left_degrees(52);

	/* Move inside the habitat, drop the animal and come back */
	stop();
	_delay_ms(100);
	if(b==1)		//If 2 animals have the same habitat
	    forward_mm(55);
	else
	    forward_mm(60);
	drop();
	if(b==1)
	   back_mm(60);
	else
	   back_mm(70);
	stop();
	_delay_ms(105);
	if(s==1)
	{  
		left_degrees(35);	
	}		
	else
		right_degrees(37);
	stop();
	_delay_ms(50);
	back_mm(21);
	stop();
	_delay_ms(50);
	q=0;		//Make node-count flag to 0 after each drop
	return;
}

/* Function to read the white line sensors' values and follow the black line for given number of nodes (nodes) */
void nodetraverse(int nodes)
{
	/*
	Function Name: nodetraverse
	Input: nodes
	Output: None
	*/
	int node_cnt=0,cond_flag=0;
	init_devices();
    while(1)
    {
		Left_white_line = ADC_Conversion(3);	//Getting data of Left WL Sensor
		Center_white_line = ADC_Conversion(2);	//Getting data of Center WL Sensor
		Right_white_line = ADC_Conversion(1);	//Getting data of Right WL Sensor
        cond_flag=0;
		//print_sensor(1,1,3);	//Prints value of White Line Sensor1
		//print_sensor(1,5,2);	//Prints Value of White Line Sensor2
		//print_sensor(1,9,1);	//Prints Value of White Line Sensor3


		// If center white line sensor is on black line
		if((Center_white_line>=0x0a) && (Center_white_line<0x78) && (Left_white_line<=0x0a) && (Right_white_line<=0x0a))
		{
		    cond_flag=1;
			forward();
			velocity(255,247);
			q++;
		}

		// If robot misses the black line completely
        if((Center_white_line<=0x0a) && (Left_white_line<=0x0a) && (Right_white_line<=0x0a))
		{
			cond_flag=1;
			forward();
			velocity(255,247);
			q++;
		}
		// If robot has deviated towards right
		if( cond_flag==0 && (Left_white_line>0x0a) && (Right_white_line<=0x0a)&&(Center_white_line<=0x0a))
		{
			cond_flag=1;
			soft_left_degrees(2);
		}

		// If robot has deviated towards left
		if(cond_flag==0 && (Right_white_line>0x0a) && (Left_white_line<=0x0a) && (Center_white_line<=0x0a))
		{
			cond_flag=1;
		    soft_right_degrees(2);
		}

		// If robot passes through a node
		if(cond_flag==0 && (Center_white_line>=0x82 || Right_white_line>=0x82 || Left_white_line>=0x82))
		{
		    cond_flag=1;
		    if(node_cnt<nodes)
			{
		       forward();
		       velocity(255,247);
			}
			if(q>=20)
			{
			    node_cnt++;
				q=0;
				if(node_cnt==nodes) // If the robot successfully traversed given number of nodes
					break;
			}
		}
	}
	stop();
	forward();
	velocity(0,0);
	return;
}

/* Function to pick the animal using arm */
void pick()
{
	/*
	Function Name: pick
	Input: None
	Output: None
	*/
	init_devices1();
	unsigned char i = 0;
	
	// Open the grip
	servo_1(10);
	
	_delay_ms(500);
	servo_1_free();
	servo_2_free();
	
	// Bend the arm down towards the animal
	servo_2(42);
	
	_delay_ms(500);
	servo_1_free();
	servo_2_free();
	
	// Pick up the animal by closing the grip
	servo_1(18);

	_delay_ms(500);
	
	// Lift the animal by rotating the arm upwards
	servo_2(5);
	_delay_ms(500);
}

/* Function to drop the animal at the habitat using arm */
void drop()
{
	/*
	Function Name: drop
	Input: None
	Output: None
	*/
	init_devices1();
	unsigned char i = 0;
	servo_1(19);
	_delay_ms(400);
	// Bend the arm towards the habitat
	for (i = 0; i < 41; i++)
	{
		servo_2(i);
		_delay_ms(15);
	}
	_delay_ms(500);
	servo_2_free();
	
	// Open the grip and drop the animal
	for (i = 0; i < 10; i++)
	{
		servo_1(i);
		_delay_ms(15);
	}
	_delay_ms(500);
	
	// Lift the arm back
	servo_2(5);
	_delay_ms(500);
	servo_1_free();
	servo_2_free();
	
	// Close the grip
	servo_1(19);
	servo_1_free();
	servo_2_free();
	return;
}


//Main Function
int main()
{
	/*
	Function Name: main
	Input: None
	Output: None
	*/
	init_devices();
	int len=0,h=0,c=0,n,rx,ry,*temp,x1,y1,m=0,x=0,y=0;
	char* locations, *a;
	char pos[2];
	while(1)
	{
		if((PINE & 0x80) == 0x80) // If boot switch is not pressed
		{  
			lcd_port_config();
			lcd_set_4bit();
			lcd_init();
			len=strlen(serial_ip);
			if(serial_ip[len-1]=='#') // To check for end of serial input
			{
				/* Extract animal and habitat locations and store them in "processed_ip" */
				h=0;
				for(i=0;i<strlen(serial_ip);i++)
				{
					if(serial_ip[i]!='\\' && serial_ip[i]!='n' && serial_ip[i]!='#')
						processed_ip[h++]=serial_ip[i];
				}
				locations=strtok(processed_ip,", ");
				while(locations!=NULL)
				{
					a=mapping[count];
					strcpy(a,locations);
					locations=strtok(NULL,", ");
					a++;
					count++;
				}

				/* Print the received data on LCD */

				// Print habitat locations
				for(int e=0;e<count/2;e++)
				{
					lcd_string(mapping[e]);
					if(e<(count/2)-1)
					   lcd_string(", ");
			    }
				lcd_cursor(2,1);

				// Print animal locations
				for(int e1=count/2;e1<count;e1++)
				{
					lcd_string(mapping[e1]);
					if(e1<(count-1))
					   lcd_string(", ");
				}
				break;
			}
		}
	}

    

	while(1)
	{	
		
		if((PINE & 0x80) != 0x80) // If boot switch is pressed
		{	    init_devices();
			    c=0;
				m=count/2;
				while(1)
				{   
					Left_white_line = ADC_Conversion(3);	//Getting data of Left WL Sensor
					Center_white_line = ADC_Conversion(2);	//Getting data of Center WL Sensor
					Right_white_line = ADC_Conversion(1);	//Getting data of Right WL Sensor

					if(Center_white_line>0x70 || Left_white_line>0x70 || Right_white_line>0x70 )
					{
						forward();
						velocity(0,0);
						break;
					}
					else
					{
						forward();
						velocity(150,150);
					}
				}
				init_devices();
                forward_mm(5);
				stop();
				_delay_ms(50);
				while(c<count/2)
				{
					/* Extract individual animal and habitat locations & call functions for each pair to move them to their respective locations */
					n=atoi(mapping[c]);
					init_devices();
					strcpy(pos, mapping[m++]);
					x=pos[0]-65;
					y=pos[1]-'1';
					temp=nodes(n-1,x,y);
					rx=temp[0];
					ry=temp[1];
					if(c==0) // Move from start node to first animal
						path(0,0,x,y);
					else // Move from a given habitat to the next animal
						path(x1,y1,x,y);

					anipick(pos); // Pick the animal
					path(x,y,rx,ry); // Move to its habitat
					habidrop(n-1,rx,ry); // Drop the animal

					// Update the current position of the robot
					x1=rx;
					y1=ry;
					c++;
				}

			   // Sound buzzer to indicate completion of task
			   buzzer_on();
			   _delay_ms(5000);
			   buzzer_off();
			   break;
		}
	}
	return;
}
