/************************************************** * * * First attempt at a code to calcule lost barley * * by A. Farmer * * 18/05/18 * * * **************************************************/

// Include any headers from the C standard library here
#include <stdio.h>

// Define any constants that I need to use here
const float PI = 3.1415;

// This is where I should put my function prototypes
float area_of_circle(float radius); 
float area_of_ring(float radius, float thickness);
float area_of_rectangle(float length, float breadth);
float lost_percent(float lost_area, float field_area);

// Now I start my code with main()
int main() {

    // In here I need to declare my variables
	int i = 0; // loop counter
	int shape = 1; // shape of the lost barley. default 1 = circle
	int nShapes = 5; // number of shapes left by aliens
	float length; // length of rectangular field
	float breadth; // breadth of rectangular field
	float field_area; // total area of rectangular field
	float radius[nShapes]; // array to store radii
	float area[nShapes]; // array to store areas
	float total_area = 0; // total area lost
	float percentage_lost = 0; // percentage of total area lost compared to field area
	float loss_in_kg; // total barley lost in kg
	float loss_in_pounds; // total barley lost in pounds

    // Next I need to get input from the user.

    // Ask for dimensions of field
	printf("Please input the dimensions of the rectangular field as MxN, whereby M and N are length and breadth: ");
	scanf("%fx%f", &length, &breadth);
	field_area = area_of_rectangle(length, breadth);

	while (i < nShapes) {
		// Ask user for shape of alien residue
		// Assume for now shape is always circular
		printf("Select lost barley shape (as a number):\n");
		printf("1. Circles\n");
		printf("2. Rings\n");

		// With error checking
		while (1) {
			scanf("%i", &shape);
			if ((shape != 1) && (shape != 2)) {
				printf("Please input a valid number from the above list!\n");
			} else {
				break;
			}
		}
		// Ask the user to input the radii.
		printf("Input the radius: ");
		scanf("%f", &radius[i]);

		// Calculate the correct area based on shape
		if (shape == 1) {
		area[i] = area_of_circle(radius[i]);
		} else {
		float thickness;
		printf("Input the thickness: ");
		scanf("%f", &thickness);
		area[i] = area_of_ring(radius[i], thickness);
		}
		total_area += area[i];

		i++;
	}

	percentage_lost = lost_percent(total_area, field_area);
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
    printf("In percentage this is %.2f %\n", percentage_lost);
    printf("Total loss in kg is:\t\t%f kg\n", loss_in_kg);
    printf("Total loss in pounds is:\t£ %f\n", loss_in_pounds);

    return(0);
}

// I'll put my functions here:

float area_of_ring(float radius, float thickness) {
	extern const float PI;
	float area;

	area = 2 * PI * radius * thickness;

	return area;
}

float area_of_circle(float radius) {
	extern const float PI;
	float area;

	area = PI * radius * radius;

	return area;
}

float area_of_rectangle(float length, float breadth) {
	float area;

	area = length * breadth;

	return area;
}

float lost_percent(float lost_area, float field_area) {
	float percent;

	percent = (lost_area / field_area) * 100;

	return percent;
}
