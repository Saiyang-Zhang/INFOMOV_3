#pragma once

#define EPSILON		0.0001f
#define MAXDEPTH	7 // live wild

namespace Tmpl8
{

class Renderer : public TheApp
{
public:
	// game flow methods
	void Init();
	float3 Trace( Ray& ray);
	float3 DirectIllumination( const float3& I, const float3& N );
	void Tick( float deltaTime );
	void UI();
	void Shutdown()
	{		
		FILE* f = fopen( "appstate.dat", "wb" ); // serialize cam
		fwrite( &camera, 1, sizeof( Camera ), f );
	}
	// input handling
	void MouseUp( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseDown( int button ) { /* implement if you want to detect mouse button presses */ }
	void MouseMove( int x, int y ) { mousePos.x = x, mousePos.y = y; }
	void MouseWheel( float y ) { /* implement if you want to handle the mouse wheel */ }
	void KeyUp( int key ) { /* implement if you want to handle keys */ }
	void KeyDown( int key ) { /* implement if you want to handle keys */ }
	// data members
	int2 mousePos;
	Scene scene;
	Camera camera;
	bool animating = true;
	float anim_time = 0;
	// fps smoothing
	float avg = 10, alpha = 1;

	Ray GetRay(const float3 origin, const float3 direction, float3 rate, int depth, const float distance, const int idx);
	Ray GetPrimaryRay(float x, float y, float3 rate, int depth, float* camera_params);

	float camera_params[12];
	uint accumulator[SCRWIDTH * SCRHEIGHT];

	Kernel* kernel_trace = nullptr;

	Buffer* buffer_accumulator = nullptr;
	Buffer* buffer_camera = nullptr;
	Buffer* buffer_logo = nullptr;
	Buffer* buffer_red = nullptr;
	Buffer* buffer_blue = nullptr;
};

} // namespace Tmpl8