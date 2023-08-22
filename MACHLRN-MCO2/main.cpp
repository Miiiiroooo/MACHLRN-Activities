#include "pch.h"
#include "USLProgram.h"


int main()
{
	srand((unsigned)time(0));

	USLProgram program = USLProgram();
	program.Initialize("files/InternetSurveyDataset.csv");
	program.RunProgram();

	return 0;
}