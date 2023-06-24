#define SCRWIDTH	1024
#define SCRHEIGHT	640
#define INFTY 1e34f

float3 camPos, topLeft, topRight, bottomLeft;
struct Surface logo, red, blue;

struct test
{
	float x, y;
};

__kernel void test(__global float* a)
{
	int idx = get_global_id( 0 );
	int idy = get_global_id( 1 );
}

struct Ray
{
	// ray data
	float3 O, D, rD;
	float t;
	int objIdx;
	bool inside; // true when in medium
};

struct Ray GetRay(const float3 origin, const float3 direction, const float distance, const int idx)
{
	struct Ray ray;
	ray.O = origin;
	ray.D = direction;
	ray.t = distance;
	ray.objIdx = idx;
	ray.inside = false;
	ray.rD.x = 1 / ray.D.x, ray.rD.y = 1 / ray.D.y, ray.rD.z = 1 / ray.D.z;
	return ray;
}

struct Ray GetPrimaryRay(float x, float y) 
{
	// calculate pixel position on virtual screen plane
	const float u = (float)x * (1.0f / SCRWIDTH);
	const float v = (float)y * (1.0f / SCRHEIGHT);
	const float3 P = topLeft + u * (topRight - topLeft) + v * (bottomLeft - topLeft);

	return GetRay(camPos, normalize( P - camPos ), INFTY, -1);
}

struct Sphere
{
	float3 pos;
	float r2, invr;
	int objIdx;
};
struct Sphere GetSphere( int idx, float3 p, float r )
	{
		struct Sphere sphere;
		sphere.pos = p;
		sphere.r2 = r* r;
		sphere.invr = 1 / r;
		sphere.objIdx = idx;
		return sphere;
	}
void IntersectSphere(struct Sphere sphere, struct Ray ray)
{
	float3 oc = ray.O - sphere.pos;
	float b = dot(oc, ray.D);
	float c = dot(oc, oc) - sphere.r2;
	float t, d = b * b - c;
	if (d <= 0) return;
	d = sqrtf(d), t = -b - d;
	bool hit = t < ray.t && t > 0;
	if (hit)
	{
		ray.t = t, ray.objIdx = sphere.objIdx;
		return;
	}
	if (c > 0) return; // we're outside; safe to skip option 2
	t = d - b, hit = t < ray.t && t > 0;
	if (hit) ray.t = t, ray.objIdx = sphere.objIdx;
}
bool IsOccludedSphere(struct Sphere sphere, struct Ray ray)
{
	float3 oc = ray.O - sphere.pos;
	float b = dot(oc, ray.D);
	float c = dot(oc, oc) - sphere.r2;
	float t, d = b * b - c;
	if (d <= 0) return false;
	d = sqrtf(d), t = -b - d;
	bool hit = t < ray.t && t > 0;
	return hit;
}
float3 GetNormalSphere(struct Sphere sphere, float3 I)
{
	return (I - sphere.pos) * sphere.invr;
}
float3 GetAlbedoSphere(float3 I)
{
	return 0.93f;
}


struct Surface
{
	uint pixels[300000];
};
struct Plane
{
	float3 N;
	float d;
	int objIdx;
};
struct Plane GetPlane( int idx, float3 normal, float dist )
{
	struct Plane plane;
	plane.N = normal;
	plane.d = dist;
	plane.objIdx = idx;
}
void IntersectPlane(struct Plane plane, struct Ray ray)
{
	float t = -(dot(ray.O, plane.N) + plane.d) / (dot(ray.D, plane.N));
	if (t < ray.t && t > 0) ray.t = t, ray.objIdx = plane.objIdx;
}
float3 GetNormalPlane(struct Plane plane, float3 I)
{
	return plane.N;
}
float3 GetAlbedoPlane(struct Plane plane, float3 I)
{
	if (plane.N.y == 1)
	{
		// floor albedo: checkerboard
		int ix = (int)(I.x * 2 + 96.01f);
		int iz = (int)(I.z * 2 + 96.01f);
		// add deliberate aliasing to two tile
		if (ix == 98 && iz == 98) ix = (int)(I.x * 32.01f), iz = (int)(I.z * 32.01f);
		if (ix == 94 && iz == 98) ix = (int)(I.x * 64.01f), iz = (int)(I.z * 64.01f);
		float3 temp = ((ix + iz) & 1) ? 1 : 0.3f;
		return temp;
	}
	else if (plane.N.z == -1)
	{
		// back wall: logo
		int ix = (int)((I.x + 4) * (128.0f / 8)), iy = (int)((2 - I.y) * (64.0f / 3));
		uint p = logo.pixels[(ix & 127) + (iy & 63) * 128];
		float3 i3;
		i3.x = (p >> 16) & 255;
		i3.y = (p >> 8) & 255;
		i3.z = p & 255;
		return i3 * (1.0f / 255.0f);
	}
	else if (plane.N.x == 1)
	{
		// left wall: red
		int ix = (int)((I.z - 4) * (512.0f / 7)), iy = (int)((2 - I.y) * (512.0f / 3));
		uint p = red.pixels[(ix & 511) + (iy & 511) * 512];
		float3 i3;
		i3.x = (p >> 16) & 255;
		i3.y = (p >> 8) & 255;
		i3.z = p & 255;
		return i3 * (1.0f / 255.0f);
	}
	else if (plane.N.x == -1)
	{
		// right wall: blue
		int ix = (int)((I.z - 4) * (512.0f / 7)), iy = (int)((2 - I.y) * (512.0f / 3));
		uint p = blue.pixels[(ix & 511) + (iy & 511) * 512];
		float3 i3;
		i3.x = (p >> 16) & 255;
		i3.y = (p >> 8) & 255;
		i3.z = p & 255;
		return i3 * (1.0f / 255.0f);
	}
	return 0.93;
}

struct Cube
{
	float3 b[2];
	int objIdx;
};


__kernel void Trace(float t, __global uint* accumulator,  __global float* camera_params, __global uint* pixels_logo, __global uint* pixels_red, __global uint* pixels_blue)
{
	camPos.x = camera_params[0], camPos.y = camera_params[1], camPos.z = camera_params[2];
	topLeft.x = camera_params[3], topLeft.y = camera_params[4], topLeft.z = camera_params[5];
	topRight.x = camera_params[6], topRight.y = camera_params[7], topRight.z = camera_params[8];
	bottomLeft.x = camera_params[9], bottomLeft.y = camera_params[10], bottomLeft.z = camera_params[11];

	int i;
	uint size_logo = sizeof( pixels_logo ), size_red = sizeof( pixels_red ), size_blue = sizeof( pixels_blue );
	for(i = 0; i < size_logo; i++) logo.pixels[i] = pixels_logo[i];
	for(i = 0; i < size_red; i++) red.pixels[i] = pixels_red[i];
	for(i = 0; i < size_blue; i++) blue.pixels[i] = pixels_blue[i];
	 
	int idx = get_global_id( 0 );
	int idy = get_global_id( 1 );

	struct Ray ray = GetPrimaryRay(idx, idy);


}