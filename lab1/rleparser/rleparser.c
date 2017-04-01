#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>


char** parseInput(char *filename, int* x_value, int* y_value) {
	FILE *ifp;
	char **cellTable;

	ifp = fopen(filename, "r");

	if (ifp == NULL) {
		printf("error: cannot open file: %s\n", filename);
		exit(-1);
	}


	int c;

	int counter = 0;

	int i, j;


	char firstline[256];
	while (fgets(firstline, 256, ifp) != NULL) {
		if (firstline[0] == '#')
			continue;
		else { 
			sscanf(firstline, "x = %d, y = %d", x_value, y_value);
			break;
		}
	}
	printf ("parsing... (x = %d, y = %d)\n", *x_value, *y_value);

	cellTable = malloc(*y_value * sizeof(char*));
	for (i = 0; i < *y_value; i++)
		cellTable[i] = malloc(*x_value * sizeof(char));
	
	int x_count = 0, y_count = 0;

	c = fgetc(ifp);
	while ((char)c != '!') {
		//printf("%c ", c);
		if (isdigit(c)) {
			counter *= 10;
			counter += (int)((char)c - '0');
		}
		else if (c == 'b') {
			if (counter == 0)
				counter = 1;
			for (i = 0; i < counter; i++){
				cellTable[y_count][x_count + i] = 0;
			}
			x_count += counter;
			counter = 0;
		}
		else if (c == 'o') {
			if (counter == 0)
				counter = 1;
			for (i = 0; i < counter; i++) {
				cellTable[y_count][x_count + i] = 1;
			}
			x_count += counter;
			counter = 0;
		}
		else if (c == '$') {
			// put 0 in implicitly defined dead cell
			if (x_count < *x_value) {
				for (i = x_count; i < *x_value; i++) {
					cellTable[y_count][i] = 0;
				}
			}
			counter = 0;
			x_count = 0;
			y_count++;
		}
		c = fgetc(ifp);
	}
	if (x_count < *x_value) {
		for (i = x_count; i < *x_value; i++) {
			cellTable[y_count][i] = 0;
		}
	};
	counter = 0;
	x_count = 0;
	y_count++;


	if (y_count < *y_value) {
		for (i = y_count; i < *y_value; i++) {
			for (j = 0; j < *x_value; j++) {
				cellTable[i][j] = 0;
			}
		}
	}



	fclose(ifp);;

	return cellTable;
}


void writeOutput(char* filename, char** cellTable, int* x_value, int* y_value, 
	const int transpose, const int customwh, const int customW, const int customH) {

	int c;
	int x_count = 0, y_count = 0;
	int i, j;


	FILE *rfp;
	

	rfp = fopen(filename, "w");

	if (rfp == NULL) {
		printf("error: cannot open file: %s\n", filename);
		exit(-1);
	}

	if (customwh) {
		if (!transpose) {
			if ((customW - *x_value) < 0 || (customH - *y_value) < 0)
				printf("warning: video size smaller than bounding box size, offsets will not be output\n");
			else {
				fprintf(rfp, "static const int offsetX = %d;\n",  (customW - *x_value)/2);
				fprintf(rfp, "static const int offsetY = %d;\n\n", (customH - *y_value)/2);
			}
		}
		else {
			if ((customW - *y_value) < 0 || (customH - *x_value) < 0)
				printf("warning: video size smaller than bounding box size, offsets will not be output\n");
			else {
				fprintf(rfp, "static const int offsetX = %d;\n",  (customW - *y_value)/2);
				fprintf(rfp, "static const int offsetY = %d;\n\n", (customH - *x_value)/2);
			}
		}
	}


	if (!transpose) {
		fprintf(rfp, "static const int tableX = %d;\n", *x_value);
		fprintf(rfp, "static const int tableY = %d;\n\n", *y_value);
		fprintf(rfp, "__device__ static const bool initTable[%d][%d] = {\n{", *y_value, *x_value);
		for (i = 0; i < *y_value; i++) {
			for (j = 0; j < *x_value; j++) {
				if (cellTable[i][j] == 1) {
					fprintf(rfp, "true");
					if (j < *x_value - 1)
						fprintf(rfp, ", ");
				}
				else {
					fprintf(rfp, "false");
					if (j < *x_value - 1)
						fprintf(rfp, ", ");
				}
			}
			fprintf(rfp, "}");
			if (i < *y_value - 1)
				fprintf(rfp, ",\n{");
		}
		fprintf(rfp, "};");
	} else {
		fprintf(rfp, "static const int tableX = %d;\n", *y_value);
		fprintf(rfp, "static const int tableY = %d;\n\n", *x_value);
		fprintf(rfp, "__device__ static const bool initTable[%d][%d] = {\n{", *x_value, *y_value);
		for (i = 0; i < *x_value; i++) {
			for (j = 0; j < *y_value; j++) {
				if (cellTable[j][i] == 1) {
					fprintf(rfp, "true");
					if (j < *y_value - 1)
						fprintf(rfp, ", ");
				}
				else {
					fprintf(rfp, "false");
					if (j < *y_value - 1)
						fprintf(rfp, ", ");
				}
			}
			fprintf(rfp, "}");
			if (i < *x_value - 1)
				fprintf(rfp, ",\n{");
		}
		fprintf(rfp, "};");
	}


	fclose(rfp);
}

void printHelp(char *argv0) {
	printf("usage: %s [-h] [-i inputfile] [-o outputfile] [-p videoWidth videoHeight cellScale] [-t]\n\n", argv0);
	printf("description:\n");
	printf("\tThis is a .rle format parser. It will parse the .rle format (which describes the initial pattern in the Conway's Game of life) and output the CUDA-compatible code (array) that can be run in \"lab1.cu\".\n\tThis program is also a (minor) part of my lab 1 homework in the GPGPU Spring 2017 course, held in CSIE, NTU.\n\n");
	printf("flags:\n");
	printf("\t-h:\tprint help and exit\n");
	printf("\t-i:\tuse custom filename as input, otherwise use \"input.rle\"\n");
	printf("\t-o:\tuse custom filename as output, otherwise use \"output\"\n");
	printf("\t-p:\tauto-calculate offsetX and offsetY by giving the width, height and scale (e.g. scale = 2 => cell = 2 x 2 = 4 pixels) of the video\n");
	printf("\t-t:\ttranspose the output matrix\n\n");
	printf("note:\n");
	printf("\tMore .rle files can be downloaded in \"http://www.conwaylife.com/wiki/Main_Page\".\n");
}



int main(int argc, char **argv) {

	int help = 0, customfilename = 0, customoutput = 0, customwh = 0, transpose = 0;
	char filename[64], outputfilename[64];
	int i, j;
	int customW = 0, customH = 0, customScale = 1;
	for (i = 1; i < argc; i++) {
		if ( strcmp(argv[i], "-h") == 0)
			help = 1;
		else if (strcmp(argv[i], "-i") == 0) {
			i++;
			if (i < argc) {
				strcpy(filename, argv[i]);
				customfilename = 1;
			}
			else
				help = 1;
		}
		else if (strcmp(argv[i], "-o") == 0) {
			i++;
			if (i < argc) {
				strcpy(outputfilename, argv[i]);
				customoutput = 1;
			}
			else
				help = 1;
		}
		else if (strcmp(argv[i], "-p") == 0) {
			i += 3;
			if (i < argc) {
				customW = atoi(argv[i - 2]);
				customH = atoi(argv[i - 1]);
				customScale = atoi(argv[i]);
				if (customScale <= 0) {
					printf("error: cellScale must be positive\n");
					exit(-1);
				}
				customwh = 1;
			}
			else {
				help = 1;
			}
		}
		else if (strcmp(argv[i], "-t") == 0) {
			transpose = 1;
		}
	}

	if (help == 1) {
		printHelp(argv[0]);
		return 0;
	}

	int *x_value;
	int *y_value;
	char **cellTable;

	x_value = malloc(sizeof(int));
	y_value = malloc(sizeof(int));

	*x_value = 0;
	*y_value = 0;


	if (customfilename) {
		cellTable = parseInput(filename, x_value, y_value);
	}
	else {
		cellTable = parseInput("input.rle", x_value, y_value);
	}

#if 0
	for (i = 0; i < *y_value; i++) {
		for (j = 0; j < *x_value; j++) {
			printf("%d", cellTable[i][j]);
		}
		printf("\n");
	}
#endif
	if (customoutput)
		writeOutput(outputfilename, cellTable, x_value, y_value, transpose, customwh, customW / customScale, customH / customScale);
	else
		writeOutput("output", cellTable, x_value, y_value, transpose, customwh, customW / customScale, customH / customScale);




	for (i = 0; i < *y_value; i++)
		free(cellTable[i]);
	free(cellTable);

	free(x_value);
	free(y_value);


	
	return 0;
}