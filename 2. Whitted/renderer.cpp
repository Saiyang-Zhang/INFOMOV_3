#include "precomp.h"

Ray Renderer::GetRay(const float3 origin, const float3 direction, float3 rate, int depth, const float distance = 1e34f, const int idx = -1)
{
	Ray ray;
	ray.O = origin;
	ray.D = direction;
	ray.t = distance;
	ray.objIdx = idx;
	ray.inside = false;
	ray.rD.x = 1 / ray.D.x, ray.rD.y = 1 / ray.D.y, ray.rD.z = 1 / ray.D.z;
	ray.rate = rate, ray.depth = depth;
	return ray;
}

Ray Renderer::GetPrimaryRay(float x, float y, float3 rate, int depth, float* camera_params)
{
	float3 camPos, topLeft, topRight, bottomLeft;
	camPos.x = camera_params[0], camPos.y = camera_params[1], camPos.z = camera_params[2];
	topLeft.x = camera_params[3], topLeft.y = camera_params[4], topLeft.z = camera_params[5];
	topRight.x = camera_params[6], topRight.y = camera_params[7], topRight.z = camera_params[8];
	bottomLeft.x = camera_params[9], bottomLeft.y = camera_params[10], bottomLeft.z = camera_params[11];

	// calculate pixel position on virtual screen plane
	const float u = (float)x * (1.0f / SCRWIDTH);
	const float v = (float)y * (1.0f / SCRHEIGHT);
	const float3 P = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);

	return GetRay(camPos, normalize(P - camPos), rate, depth, 1e34f, -1);
}

// -----------------------------------------------------------
// Initialize the renderer
// -----------------------------------------------------------
void Renderer::Init()
{
	// retrieve cam
	FILE* f = fopen( "appstate.dat", "rb" );
	if (f)
	{
		fread( &camera, 1, sizeof( Camera ), f );
		fclose( f );
	}

	kernel_trace = new Kernel("kernel.cl", "Trace");

	static Surface logo("../assets/logo.png");
	static Surface red("../assets/red.png");
	static Surface blue("../assets/blue.png");
	buffer_logo = new Buffer(sizeof(uint) * logo.width * logo.height, logo.pixels, CL_MEM_READ_ONLY);
	buffer_red = new Buffer(sizeof(uint) * red.width * red.height, red.pixels, CL_MEM_READ_ONLY);
	buffer_blue = new Buffer(sizeof(uint) * blue.width * blue.height, blue.pixels, CL_MEM_READ_ONLY);

	buffer_logo->CopyToDevice();
	buffer_red->CopyToDevice();
	buffer_blue->CopyToDevice();
}

// -----------------------------------------------------------
// Gather direct illumination for a point
// -----------------------------------------------------------
float3 Renderer::DirectIllumination( const float3& I, const float3& N )
{
	// sum irradiance from light sources
	float3 irradiance( 0 );
	// query the (only) scene light
	float3 pointOnLight = scene.GetLightPos();
	float3 L = pointOnLight - I;
	float distance = length( L );
	L *= 1 / distance;
	float ndotl = dot( N, L );
	if (ndotl < EPSILON) /* we don't face the light */ return 0;
	// cast a shadow ray
	struct Ray s = GetRay( I + L * EPSILON, L, distance - 2 * EPSILON, 1, 0);
	if (!scene.IsOccluded( s ))
	{
		// light is visible; calculate irradiance (= projected radiance)
		float attenuation = 1 / (distance * distance);
		float3 in_radiance = scene.GetLightColor() * attenuation;
		irradiance = in_radiance * dot( N, L );
	}
	return irradiance;
}

// -----------------------------------------------------------
// Evaluate light transport
// -----------------------------------------------------------
float3 Renderer::Trace(Ray& primaryRay)
{
	float3 out_radiance = 0;

	struct Ray rays[128], ray;
	int cur = 0, top = 1;
	rays[0] = primaryRay;

	while (cur != top) {
		ray = rays[cur++];

		// intersect the ray with the scene
		scene.FindNearest(ray);
		if (ray.objIdx == -1) /* ray left the scene */ continue;
		if (ray.depth > MAXDEPTH) /* bouned too many times */ continue;
		// gather shading data
		float3 I = ray.O + ray.t * ray.D;
		float3 N = scene.GetNormal(ray.objIdx, I, ray.D);
		float3 albedo = scene.GetAlbedo(ray.objIdx, I);
		// do whitted
		
		float reflectivity = scene.GetReflectivity(ray.objIdx, I);
		float refractivity = scene.GetRefractivity(ray.objIdx, I);
		float diffuseness = 1 - (reflectivity + refractivity);
		// handle pure speculars such as mirrors
		if (reflectivity > 0)
		{
			float3 R = reflect(ray.D, N);
			struct Ray r = GetRay(I + R * EPSILON, R, reflectivity * albedo * ray.rate, ray.depth + 1);
			rays[top++] = r;
		}
		// handle dielectrics such as glass / water
		if (refractivity > 0)
		{
			float3 medium_scale(1);
			if (ray.inside)
			{
				float3 absorption = float3(0.5f, 0, 0.5f); // scene.GetAbsorption( objIdx );
				medium_scale.x = expf(absorption.x * -ray.t);
				medium_scale.y = expf(absorption.y * -ray.t);
				medium_scale.z = expf(absorption.z * -ray.t);
			}
			float3 R = reflect(ray.D, N);
			float n1 = ray.inside ? 1.2f : 1, n2 = ray.inside ? 1 : 1.2f;
			float eta = n1 / n2, cosi = dot(-ray.D, N);
			float cost2 = 1.0f - eta * eta * (1 - cosi * cosi);
			float Fr = 1;
			if (cost2 > 0)
			{
				float a = n1 - n2, b = n1 + n2, R0 = (a * a) / (b * b), c = 1 - cosi;
				Fr = R0 + (1 - R0) * (c * c * c * c * c);
				float3 T = eta * ray.D + ((eta * cosi - sqrtf(fabs(cost2))) * N);
				struct Ray t = GetRay(I + T * EPSILON, T, medium_scale * albedo * (1 - Fr) * ray.rate, ray.depth + 1);
				t.inside = !ray.inside;
				rays[top++] = t;
			}
			struct Ray r = GetRay(I + R * EPSILON, R, albedo * Fr * ray.rate, ray.depth+1);
			rays[top++] = r;
		}
		// handle diffuse surfaces
		if (diffuseness > 0)
		{
			// calculate illumination
			float3 irradiance = DirectIllumination(I, N);
			// we don't account for diffuse interreflections: approximate
			float3 ambient = float3(0.2f, 0.2f, 0.2f);
			// calculate reflected radiance using Lambert brdf
			float3 brdf = albedo * INVPI;
			out_radiance += diffuseness * brdf * (irradiance + ambient) * ray.rate;
		}
		// apply absorption if we travelled through a medium
	}

	return out_radiance;
}


// -----------------------------------------------------------
// Main application tick function - Executed once per frame
// -----------------------------------------------------------
void Renderer::Tick( float deltaTime )
{
	int global_size = SCRWIDTH * SCRHEIGHT;
	int local_size = 128;

	camera_params[0] = camera.camPos.x, camera_params[1] = camera.camPos.y, camera_params[2] = camera.camPos.z;
	camera_params[3] = camera.topLeft.x, camera_params[4] = camera.topLeft.y, camera_params[5] = camera.topLeft.z;
	camera_params[6] = camera.topRight.x, camera_params[7] = camera.topRight.y, camera_params[8] = camera.topRight.z;
	camera_params[9] = camera.bottomLeft.x, camera_params[10] = camera.bottomLeft.y, camera_params[11] = camera.bottomLeft.z;

	buffer_camera = new Buffer(sizeof( camera_params ), camera_params, CL_MEM_READ_ONLY);
	buffer_accumulator = new Buffer(sizeof(accumulator), accumulator, CL_MEM_READ_WRITE);

	buffer_camera->CopyToDevice();
	buffer_accumulator->CopyToDevice();
	kernel_trace->SetArguments(anim_time, buffer_accumulator, buffer_camera, buffer_logo, buffer_red, buffer_blue);
	kernel_trace->Run(global_size, local_size);
	buffer_accumulator->CopyFromDevice();

	// animation
	if (animating) scene.SetTime( anim_time += deltaTime * 0.002f );
	// pixel loop
	Timer t;
	// lines are executed as OpenMP parallel tasks (disabled in DEBUG)

#pragma omp parallel for schedule(dynamic)
	for (int y = 0; y < SCRHEIGHT; y++)
		for (int dest = y * SCRWIDTH, x = 0; x < SCRWIDTH; x++)
			screen->pixels[dest + x] = accumulator[dest + x];
	// performance report - running average - ms, MRays/s
	avg = (1 - alpha) * avg + alpha * t.elapsed() * 1000;
	if (alpha > 0.05f) alpha *= 0.75f;
	// handle user input
	camera.HandleInput( deltaTime );
}

// -----------------------------------------------------------
// Update user interface (imgui)
// -----------------------------------------------------------
void Renderer::UI()
{
	// animation toggle
	ImGui::Checkbox( "Animate scene", &animating );
	// ray query on mouse
	Ray r = GetPrimaryRay( (float)mousePos.x, (float)mousePos.y, 1, 0, camera_params );
	scene.FindNearest( r );
	ImGui::Text( "Object id %i", r.objIdx );
	ImGui::Text( "Frame: %5.2fms (%.1ffps)", avg, 1000 / avg );
}