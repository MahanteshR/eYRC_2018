#include "embedded.c"

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
int turnsoft(int);
void pick();
void drop();


char processed_ip[100];
int count=0;
int f=1,b=0,r=0,l=0,a=4,p=0,q=0;
char mapping[30][30];


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


