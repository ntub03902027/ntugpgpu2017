all: 
	nvcc --std=c++11 -I../utils/ lab1.cu main.cu -o lab1
clean:
	rm -f lab1 result.y4m result.mkv
run:
	./lab1
	rm -f result.mkv
	avconv -i result.y4m result.mkv
	xplayer result.mkv
auto: all run
