# Problem statement
In this Task, teams will read these files one at a time and communicate the locations to the
robot. The robot will traverse the arena to pick the Animal blocks from given location and
deposit them in their associated Habitat. You have to place all 20 Animal blocks you had
prepared (according to Section 3.2 of Rulebook) as in Figure 5 of Rulebook (page 9). The
Animal block to be placed at all these locations is random for this Task.
#
Teams will create a Python script python_to_robot.py which will communicate serially with
Firebird V robot. First, the script will read locations from example_1.txt and later from
example_2.txt.
#
The robot receives the data on its UART character by character, hence you need to parse the
information before sending to the robot.
#
You will follow the format of serial data that you can see after reading the file. It will look as
follows for example_1.txt:
‘11, 23, 8\nC1, A5, F3’
#
To indicate the end of serial data, you will add ‘#’. So, the data you will send to the robot will
look as:
‘11, 23, 8\nC1, A5, F3 #’
