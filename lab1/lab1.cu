#include "lab1.h"
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;
#include "SyncedMemory.h"


typedef struct rgb {
	uint8_t r;
	uint8_t g;
	uint8_t b;

} RGB;

struct Lab1VideoGenerator::Impl {
	int t = 0;
	MemoryBuffer<RGB> frame;

	SyncedMemory<RGB> frame_smem;
	

	Impl() : frame(W*H) 
	{
		frame_smem = frame.CreateSync(W*H);

	}
};	



Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

__global__ void Draw(RGB *frame, int t) {
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < H && x < W) {
		frame[y*W+x].r = ( (t / 12 + x / 80 + y / 80) % 3 == 0 )? 255 : 0;
		frame[y*W+x].g = ( (t / 12 + x / 80 + y / 80) % 3 == 1 )? 255 : 0;
		frame[y*W+x].b = ( (t / 12 + x / 80 + y / 80) % 3 == 2 )? 255 : 0;
	}
}

__global__ void toYUV(const RGB *frame, uint8_t *yuv_y, uint8_t *yuv_u, uint8_t *yuv_v) {
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

// for y
	if (y < H && x < W) {
		yuv_y[y*W+x] = 0.2990f * frame[y*W+x].r + 0.5870f * frame[y*W+x].g + 0.1140f * frame[y*W+x].b;
	}

// for u, v
	if (y < H / 2 && x < W / 2) {
		yuv_u[y*W/2+x] = (-0.1690f * frame[(2*y)*W+(2*x)].r - 0.3310f * frame[(2*y)*W+(2*x)].g + 0.5000f * frame[(2*y)*W+(2*x)].b 
			- 0.1690f * frame[(2*y+1)*W+(2*x)].r - 0.3310f * frame[(2*y+1)*W+(2*x)].g + 0.5000f * frame[(2*y+1)*W+(2*x)].b) * 0.25f + 128.0;
		yuv_v[y*W/2+x] = (0.5000f * frame[(2*y)*W+(2*x)].r - 0.4190f * frame[(2*y)*W+(2*x)].g - 0.0810f * frame[(2*y)*W+(2*x)].b 
			+ 0.5000f * frame[(2*y+1)*W+(2*x)].r - 0.4190f * frame[(2*y+1)*W+(2*x)].g - 0.0810f * frame[(2*y+1)*W+(2*x)].b) * 0.25f + 128.0;
	}
}


void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	//auto frame_smem = impl->frame.CreateSync(W*H);
	
	Draw<<<dim3((W-1)/16+1,(H-1)/12+1), dim3(16,12)>>>(impl->frame_smem.get_gpu_wo(), impl->t);

#if 0
	auto test = impl->frame_smem.get_cpu_ro();

	for (int i = 0; i < H; ++i)
		for (int j = 0; j < W; ++j)
			printf("[%d, %d] : %d\n", i, j, test[i*W+j].r);
#endif


	toYUV<<<dim3((W-1)/16+1,(H-1)/12+1), dim3(16,12)>>>(impl->frame_smem.get_gpu_ro(), yuv, yuv+W*H, yuv+W*H+(W/2)*(H/2));


	




	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	//cudaMemset(yuv+W*H, 128, W*H/2);
	++(impl->t);
}
