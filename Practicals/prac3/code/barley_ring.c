/**************************************************
 *                                                *
 * First attempt at a code to calcule lost barley *
 * by A. Farmer                                   *
 * 18/05/18                                       *
 *                                                *
 **************************************************/

// Include any headers from the C standard library here
#include <stdio.h>

// Define any constants that I need to use here
const float PI = 3.1415;
const float THICKNESS = 0.25; // thickness of lost barley rings

// This is where I should put my function prototypes
float area_of_circle(float radius); 

// Now I start my code with main()
int main() {

    // In here I need to declare my variables
	int i = 0; // loop counter
	float radius[5]; // array to store radii
	float area[5]; // array to store areas
	float total_area = 0; // total area lost
	float loss_in_kg; // total barley lost in kg
	float loss_in_pounds; // total barley lost in pounds
    // Next I need to get input from the user.
    // I'll do this by using a printf() to ask the user to input the radii.
	printf("Input the radii of circles, ONE BY ONE\n");
	while (i < 5) {
		scanf("%f", &radius[i]);
		printf("\n");

		area[i] = area_of_circle(radius[i]);

		total_area += area[i];

		i++;
	}
    // Now I need to loop through the radii caluclating the area for each
	// done above
    // Next I'll sum up all of the individual areas
	// done above
    /******************************************************************
     *                                                                *
     * Now I know the total area I can use the following information: *
     *                                                                *
     * One square meter of crop produces about 135 grams of barley    *
     *                                                                *
     * One kg of barley sells for about 10 pence                      *
     *                                                                *
     ******************************************************************/

    // Using the above I'll work out how much barley has been lost.
    loss_in_kg = total_area*0.135;
    loss_in_pounds = loss_in_kg*0.10;

    // Finally I'll use a printf() to print this to the screen.
    printf("\nTotal area lost in m^2 is:\t%f m^2\n", total_area);
    printf("Total loss in kg is:\t\t%f kg\n", loss_in_kg);
    printf("Total loss in pounds is:\t£ %f\n", loss_in_pounds);

    return(0);
}

// I'll put my functions here:

float area_of_circle(float radius) {
	extern const float PI;
	extern const float THICKNESS;
	float area;

	area = 2 * PI * radius * THICKNESS;

	return area;
}
