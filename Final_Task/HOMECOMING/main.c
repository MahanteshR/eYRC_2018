/*
Team Id: HC#3266
Author List: Hemanth Kumar K.L , Aman Bhat T, Mahantesh R, Nischith B.O.
File Name: White_Line_Following.c
Theme: Homecoming(Task-4, Original configuration)
Functions (important): Nodes, Path, Anipick, Habidrop, Pick, Drop, Facing, Straight, Turn, Semiturn, Nodetraverse, Turnsoft
Global Variables: data,serial_ip,processed_ip,z,count, f,b,r,l,a,p,mapping
*/


#include "task.c"

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
