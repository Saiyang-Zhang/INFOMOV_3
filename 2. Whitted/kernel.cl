#define INFTY		1e34f
#define PI			3.14159265358979323846264f
#define SCRWIDTH	1024
#define SCRHEIGHT	640


#define FOURLIGHTS
#define USEBVH

#define PLANE_X(o,i) {float t=-(ray.O.x+o)*ray.rD.x;if(t<ray.t&&t>0)ray.t=t,ray.objIdx=i;}
#define PLANE_Y(o,i) {float t=-(ray.O.y+o)*ray.rD.y;if(t<ray.t&&t>0)ray.t=t,ray.objIdx=i;}
#define PLANE_Z(o,i) {float t=-(ray.O.z+o)*ray.rD.z;if(t<ray.t&&t>0)ray.t=t,ray.objIdx=i;}

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

struct mat4
{
	float cell[16];
};
struct mat4 Identity()
{
	struct mat4 mat;
	int i = 0;
	for(i = 0; i < 16; i++) mat.cell[i] = 0;
	for(i = 0; i < 4; i++) mat.cell[i * 5] = 1;
	return mat;
}
struct mat4 FastInvertedTransformNoScale( struct mat4 mat )
{
	struct mat4 r;
	r.cell[0] = mat.cell[0], r.cell[1] = mat.cell[4], r.cell[2] = mat.cell[8];
	r.cell[4] = mat.cell[1], r.cell[5] = mat.cell[5], r.cell[6] = mat.cell[9];
	r.cell[8] = mat.cell[2], r.cell[9] = mat.cell[6], r.cell[10] = mat.cell[10];
	
	r.cell[3] = -(mat.cell[3] * r.cell[0] + mat.cell[7] * r.cell[1] + mat.cell[11] * r.cell[2]);
	r.cell[7] = -(mat.cell[3] * r.cell[4] + mat.cell[7] * r.cell[5] + mat.cell[11] * r.cell[6]);
	r.cell[11] = -(mat.cell[3] * r.cell[8] + mat.cell[7] * r.cell[9] + mat.cell[11] * r.cell[10]);
	return r;
}
struct mat4 Translate( const float x, const float y, const float z )
{
	struct mat4 r; 
	int i = 0;
	for(i = 0; i < 16; i++) r.cell[i] = 0;
	for(i = 0; i < 4; i++) r.cell[i * 5] = 1;
	r.cell[3] = x; r.cell[7] = y; r.cell[11] = z; 
	return r;
}
struct mat4 RotateX( const float a )
{
	struct mat4 r; 
	int i = 0;
	for(i = 0; i < 16; i++) r.cell[i] = 0;
	for(i = 0; i < 4; i++) r.cell[i * 5] = 1;
	r.cell[5] = cos( a ); r.cell[6] = -sin( a ); r.cell[9] = sin( a ); r.cell[10] = cos( a ); return r;
}
struct mat4 RotateY( const float a )
{
	struct mat4 r; 
	int i = 0;
	for(i = 0; i < 16; i++) r.cell[i] = 0;
	for(i = 0; i < 4; i++) r.cell[i * 5] = 1;
	r.cell[0] = cos( a ); r.cell[2] = sin( a ); r.cell[8] = -sin( a ); r.cell[10] = cos( a ); return r;
}
struct mat4 RotateZ( const float a )
{
	struct mat4 r; 
	int i = 0;
	for(i = 0; i < 16; i++) r.cell[i] = 0;
	for(i = 0; i < 4; i++) r.cell[i * 5] = 1;
	r.cell[0] = cos( a ); r.cell[1] = -sin( a ); r.cell[4] = sin( a ); r.cell[5] = cos( a ); return r;
}
struct mat4 mat4Mul( struct mat4 a, struct mat4 b )
{
	struct mat4 r;
	for (uint i = 0; i < 16; i += 4)
		for (uint j = 0; j < 4; ++j)
		{
			r.cell[i + j] =
				(a.cell[i + 0] * b.cell[j + 0]) +
				(a.cell[i + 1] * b.cell[j + 4]) +
				(a.cell[i + 2] * b.cell[j + 8]) +
				(a.cell[i + 3] * b.cell[j + 12]);
		}
}
struct mat4 Inverted( struct mat4 a )
{
	// from MESA, via http://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
	float inv[16] = {
		a.cell[5] * a.cell[10] * a.cell[15] - a.cell[5] * a.cell[11] * a.cell[14] - a.cell[9] * a.cell[6] * a.cell[15] +
		a.cell[9] * a.cell[7] * a.cell[14] + a.cell[13] * a.cell[6] * a.cell[11] - a.cell[13] * a.cell[7] * a.cell[10],
		-a.cell[1] * a.cell[10] * a.cell[15] + a.cell[1] * a.cell[11] * a.cell[14] + a.cell[9] * a.cell[2] * a.cell[15] -
		a.cell[9] * a.cell[3] * a.cell[14] - a.cell[13] * a.cell[2] * a.cell[11] + a.cell[13] * a.cell[3] * a.cell[10],
		a.cell[1] * a.cell[6] * a.cell[15] - a.cell[1] * a.cell[7] * a.cell[14] - a.cell[5] * a.cell[2] * a.cell[15] +
		a.cell[5] * a.cell[3] * a.cell[14] + a.cell[13] * a.cell[2] * a.cell[7] - a.cell[13] * a.cell[3] * a.cell[6],
		-a.cell[1] * a.cell[6] * a.cell[11] + a.cell[1] * a.cell[7] * a.cell[10] + a.cell[5] * a.cell[2] * a.cell[11] -
		a.cell[5] * a.cell[3] * a.cell[10] - a.cell[9] * a.cell[2] * a.cell[7] + a.cell[9] * a.cell[3] * a.cell[6],
		-a.cell[4] * a.cell[10] * a.cell[15] + a.cell[4] * a.cell[11] * a.cell[14] + a.cell[8] * a.cell[6] * a.cell[15] -
		a.cell[8] * a.cell[7] * a.cell[14] - a.cell[12] * a.cell[6] * a.cell[11] + a.cell[12] * a.cell[7] * a.cell[10],
		a.cell[0] * a.cell[10] * a.cell[15] - a.cell[0] * a.cell[11] * a.cell[14] - a.cell[8] * a.cell[2] * a.cell[15] +
		a.cell[8] * a.cell[3] * a.cell[14] + a.cell[12] * a.cell[2] * a.cell[11] - a.cell[12] * a.cell[3] * a.cell[10],
		-a.cell[0] * a.cell[6] * a.cell[15] + a.cell[0] * a.cell[7] * a.cell[14] + a.cell[4] * a.cell[2] * a.cell[15] -
		a.cell[4] * a.cell[3] * a.cell[14] - a.cell[12] * a.cell[2] * a.cell[7] + a.cell[12] * a.cell[3] * a.cell[6],
		a.cell[0] * a.cell[6] * a.cell[11] - a.cell[0] * a.cell[7] * a.cell[10] - a.cell[4] * a.cell[2] * a.cell[11] +
		a.cell[4] * a.cell[3] * a.cell[10] + a.cell[8] * a.cell[2] * a.cell[7] - a.cell[8] * a.cell[3] * a.cell[6],
		a.cell[4] * a.cell[9] * a.cell[15] - a.cell[4] * a.cell[11] * a.cell[13] - a.cell[8] * a.cell[5] * a.cell[15] +
		a.cell[8] * a.cell[7] * a.cell[13] + a.cell[12] * a.cell[5] * a.cell[11] - a.cell[12] * a.cell[7] * a.cell[9],
		-a.cell[0] * a.cell[9] * a.cell[15] + a.cell[0] * a.cell[11] * a.cell[13] + a.cell[8] * a.cell[1] * a.cell[15] -
		a.cell[8] * a.cell[3] * a.cell[13] - a.cell[12] * a.cell[1] * a.cell[11] + a.cell[12] * a.cell[3] * a.cell[9],
		a.cell[0] * a.cell[5] * a.cell[15] - a.cell[0] * a.cell[7] * a.cell[13] - a.cell[4] * a.cell[1] * a.cell[15] +
		a.cell[4] * a.cell[3] * a.cell[13] + a.cell[12] * a.cell[1] * a.cell[7] - a.cell[12] * a.cell[3] * a.cell[5],
		-a.cell[0] * a.cell[5] * a.cell[11] + a.cell[0] * a.cell[7] * a.cell[9] + a.cell[4] * a.cell[1] * a.cell[11] -
		a.cell[4] * a.cell[3] * a.cell[9] - a.cell[8] * a.cell[1] * a.cell[7] + a.cell[8] * a.cell[3] * a.cell[5],
		-a.cell[4] * a.cell[9] * a.cell[14] + a.cell[4] * a.cell[10] * a.cell[13] + a.cell[8] * a.cell[5] * a.cell[14] -
		a.cell[8] * a.cell[6] * a.cell[13] - a.cell[12] * a.cell[5] * a.cell[10] + a.cell[12] * a.cell[6] * a.cell[9],
		a.cell[0] * a.cell[9] * a.cell[14] - a.cell[0] * a.cell[10] * a.cell[13] - a.cell[8] * a.cell[1] * a.cell[14] +
		a.cell[8] * a.cell[2] * a.cell[13] + a.cell[12] * a.cell[1] * a.cell[10] - a.cell[12] * a.cell[2] * a.cell[9],
		-a.cell[0] * a.cell[5] * a.cell[14] + a.cell[0] * a.cell[6] * a.cell[13] + a.cell[4] * a.cell[1] * a.cell[14] -
		a.cell[4] * a.cell[2] * a.cell[13] - a.cell[12] * a.cell[1] * a.cell[6] + a.cell[12] * a.cell[2] * a.cell[5],
		a.cell[0] * a.cell[5] * a.cell[10] - a.cell[0] * a.cell[6] * a.cell[9] - a.cell[4] * a.cell[1] * a.cell[10] +
		a.cell[4] * a.cell[2] * a.cell[9] + a.cell[8] * a.cell[1] * a.cell[6] - a.cell[8] * a.cell[2] * a.cell[5]
	};
	const float det = a.cell[0] * inv[0] + a.cell[1] * inv[4] + a.cell[2] * inv[8] + a.cell[3] * inv[12];
	struct mat4 retVal;
	if (det != 0)
	{
		const float invdet = 1.0f / det;
		for (int i = 0; i < 16; i++) retVal.cell[i] = inv[i] * invdet;
	}
	return retVal;
}

float3 TransformPosition( float3 b, struct mat4 a )
{
	float3 result;
	result.x = a.cell[0] * b.x + a.cell[1] * b.y + a.cell[2] * b.z + a.cell[3];
	result.y = a.cell[4] * b.x + a.cell[5] * b.y + a.cell[6] * b.z + a.cell[7];
	result.z = a.cell[8] * b.x + a.cell[9] * b.y + a.cell[10] * b.z + a.cell[11];
	return result;
}
float3 TransformVector( float3 b, struct mat4 a )
{
	float3 result;
	result.x = a.cell[0] * b.x + a.cell[1] * b.y + a.cell[2] * b.z;
	result.y = a.cell[4] * b.x + a.cell[5] * b.y + a.cell[6] * b.z;
	result.z = a.cell[8] * b.x + a.cell[9] * b.y + a.cell[10] * b.z;
	return result;
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
void IntersectSphere(struct Sphere *sphere, struct Ray *ray)
{
	float3 oc = ray->O - sphere->pos;
	float b = dot(oc, ray->D);
	float c = dot(oc, oc) - sphere->r2;
	float t, d = b * b - c;
	if (d <= 0) return;
	d = sqrt(d), t = -b - d;
	bool hit = t < ray->t && t > 0;
	if (hit)
	{
		ray->t = t, ray->objIdx = sphere->objIdx;
		return;
	}
	if (c > 0) return; // we're outside; safe to skip option 2
	t = d - b, hit = t < ray->t && t > 0;
	if (hit) ray->t = t, ray->objIdx = sphere->objIdx;
}
bool IsOccludedSphere(struct Sphere *sphere, struct Ray *ray)
{
	float3 oc = ray->O - sphere->pos;
	float b = dot(oc, ray->D);
	float c = dot(oc, oc) - sphere->r2;
	float t, d = b * b - c;
	if (d <= 0) return false;
	d = sqrt(d), t = -b - d;
	bool hit = t < ray->t && t > 0;
	return hit;
}
float3 GetNormalSphere(struct Sphere *sphere, float3 I)
{
	return (I - sphere->pos) * sphere->invr;
}
float3 GetAlbedoSphere( struct Sphere *sphere, float3 I)
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
void IntersectPlane(struct Plane *plane, struct Ray *ray)
{
	float t = -(dot(ray->O, plane->N) + plane->d) / (dot(ray->D, plane->N));
	if (t < ray->t && t > 0) ray->t = t, ray->objIdx = plane->objIdx;
}
float3 GetNormalPlane(struct Plane *plane, float3 I)
{
	return plane->N;
}
float3 GetAlbedoPlane(struct Plane *plane, float3 I)
{
	if (plane->N.y == 1)
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
	else if (plane->N.z == -1)
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
	else if (plane->N.x == 1)
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
	else if (plane->N.x == -1)
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
	struct mat4 M, invM;
	int objIdx;
};
struct Cube GetCube( int idx, float3 pos, float3 size, struct mat4 transform)
{
	struct Cube cube;
	cube.objIdx = idx;
	cube.b[0] = pos - 0.5f * size;
	cube.b[1] = pos + 0.5f * size;
	cube.M = transform;
	cube.invM = FastInvertedTransformNoScale( transform );
	return cube;
}
void IntersectCube( struct Cube *cube, struct Ray *ray )
{
	// 'rotate' the cube by transforming the ray into object space
	// using the inverse of the cube transform.
	float3 O = TransformPosition( ray->O, cube->invM );
	float3 D = TransformVector( ray->D, cube->invM );
	float rDx = 1 / D.x, rDy = 1 / D.y, rDz = 1 / D.z;
	int signx = D.x < 0, signy = D.y < 0, signz = D.z < 0;
	float tmin = (cube->b[signx].x - O.x) * rDx;
	float tmax = (cube->b[1 - signx].x - O.x) * rDx;
	float tymin = (cube->b[signy].y - O.y) * rDy;
	float tymax = (cube->b[1 - signy].y - O.y) * rDy;
	if (tmin > tymax || tymin > tmax) return;
	tmin = max( tmin, tymin ), tmax = min( tmax, tymax );
	float tzmin = (cube->b[signz].z - O.z) * rDz;
	float tzmax = (cube->b[1 - signz].z - O.z) * rDz;
	if (tmin > tzmax || tzmin > tmax) return;
	tmin = max( tmin, tzmin ), tmax = min( tmax, tzmax );
	if (tmin > 0)
	{
		if (tmin < ray->t) ray->t = tmin, ray->objIdx = cube->objIdx;
	}
	else if (tmax > 0)
	{
		if (tmax < ray->t) ray->t = tmax, ray->objIdx = cube->objIdx;
	}
}
bool IsOccludedCube( struct Cube *cube, struct Ray *ray )
{
	float3 O = TransformPosition( ray->O, cube->invM );
	float3 D = TransformVector( ray->D, cube->invM );
	float rDx = 1 / D.x, rDy = 1 / D.y, rDz = 1 / D.z;
	float t1 = (cube->b[0].x - O.x) * rDx, t2 = (cube->b[1].x - O.x) * rDx;
	float t3 = (cube->b[0].y - O.y) * rDy, t4 = (cube->b[1].y - O.y) * rDy;
	float t5 = (cube->b[0].z - O.z) * rDz, t6 = (cube->b[1].z - O.z) * rDz;
	float tmin = max( max( min( t1, t2 ), min( t3, t4 ) ), min( t5, t6 ) );
	float tmax = min( min( max( t1, t2 ), max( t3, t4 ) ), max( t5, t6 ) );
	return tmax > 0 && tmin < tmax&& tmin < ray->t;
}
float3 GetNormalCube( struct Cube *cube, float3 I )
{
	// transform intersection point to object space
	float3 objI = TransformPosition( I, cube->invM );
	// determine normal in object space
	float3 N = 0;
	N.x = -1;
	float d0 = fabs( objI.x - cube->b[0].x ), d1 = fabs( objI.x - cube->b[1].x );
	float d2 = fabs( objI.y - cube->b[0].y ), d3 = fabs( objI.y - cube->b[1].y );
	float d4 = fabs( objI.z - cube->b[0].z ), d5 = fabs( objI.z - cube->b[1].z );
	float minDist = d0;
	if (d1 < minDist) minDist = d1, N.x = 1;
	if (d2 < minDist){
		minDist = d2;
		N.x = 0;
		N.y = -1;
		N.z = 0;
	}
	if (d3 < minDist){
		minDist = d3;
		N.x = 0;
		N.y = 1;
		N.z = 0;
	}
	if (d4 < minDist){
		minDist = d4;
		N.x = 0;
		N.y = 0;
		N.z = -1;
	}
	if (d5 < minDist){
		minDist = d5;
		N.x = 0;
		N.y = 0;
		N.z = 1;
	}
	// return normal in world space
	return TransformVector( N, cube->M );
}
float3 GetAlbedoCube( struct Cube *cube, float3 I )
{
	return 1;
}

struct Quad
{
	float size;
	struct mat4 T, invT;
	int objIdx;
};
struct Quad GetQuad( int idx, float s, struct mat4 transform )
{
	struct Quad quad;
	quad.objIdx = idx;
	quad.size = s * 0.5f;
	quad.T = transform, quad.invT = FastInvertedTransformNoScale( transform );
	return quad;
}
void IntersectQuad( struct Quad *quad, struct Ray *ray )
{
	const float Oy = quad->invT.cell[4] * ray->O.x + quad->invT.cell[5] * ray->O.y + quad->invT.cell[6] * ray->O.z + quad->invT.cell[7];
	const float Dy = quad->invT.cell[4] * ray->D.x + quad->invT.cell[5] * ray->D.y + quad->invT.cell[6] * ray->D.z;
	const float t = Oy / -Dy;
	if (t < ray->t && t > 0)
	{
		const float Ox = quad->invT.cell[0] * ray->O.x + quad->invT.cell[1] * ray->O.y + quad->invT.cell[2] * ray->O.z + quad->invT.cell[3];
		const float Oz = quad->invT.cell[8] * ray->O.x + quad->invT.cell[9] * ray->O.y + quad->invT.cell[10] * ray->O.z + quad->invT.cell[11];
		const float Dx = quad->invT.cell[0] * ray->D.x + quad->invT.cell[1] * ray->D.y + quad->invT.cell[2] * ray->D.z;
		const float Dz = quad->invT.cell[8] * ray->D.x + quad->invT.cell[9] * ray->D.y + quad->invT.cell[10] * ray->D.z;
		const float Ix = Ox + t * Dx, Iz = Oz + t * Dz;
		if (Ix > -quad->size && Ix < quad->size && Iz > -quad->size && Iz < quad->size)
			ray->t = t, ray->objIdx = quad->objIdx;
	}
}
bool IsOccludedQuad( struct Quad *quad, struct Ray *ray )
{
	const float Oy = quad->invT.cell[4] * ray->O.x + quad->invT.cell[5] * ray->O.y + quad->invT.cell[6] * ray->O.z + quad->invT.cell[7];
	const float Dy = quad->invT.cell[4] * ray->D.x + quad->invT.cell[5] * ray->D.y + quad->invT.cell[6] * ray->D.z;
	const float t = Oy / -Dy;
	if (t < ray->t && t > 0)
	{
		const float Ox = quad->invT.cell[0] * ray->O.x + quad->invT.cell[1] * ray->O.y + quad->invT.cell[2] * ray->O.z + quad->invT.cell[3];
		const float Oz = quad->invT.cell[8] * ray->O.x + quad->invT.cell[9] * ray->O.y + quad->invT.cell[10] * ray->O.z + quad->invT.cell[11];
		const float Dx = quad->invT.cell[0] * ray->D.x + quad->invT.cell[1] * ray->D.y + quad->invT.cell[2] * ray->D.z;
		const float Dz = quad->invT.cell[8] * ray->D.x + quad->invT.cell[9] * ray->D.y + quad->invT.cell[10] * ray->D.z;
		const float Ix = Ox + t * Dx, Iz = Oz + t * Dz;
		return Ix > -quad->size && Ix < quad->size&& Iz > -quad->size && Iz < quad->size;
	}
	return false;
}
float3 GetNormalQuad( struct Quad *quad, float3 I )
{
	// TransformVector( float3( 0, -1, 0 ), T ) 
	float3 n;
	n.x = -quad->T.cell[1];
	n.y = -quad->T.cell[5];
	n.z = -quad->T.cell[9];
	return n;
}
float3 GetAlbedoQuad( struct Quad *quad, float3 I )
{
	return 10;
}

struct Torus
{
	float rt2, rc2, r2;
	int objIdx;
	struct mat4 T, invT;
};
struct Torus GetTorus( int idx, float a, float b)
{
	struct Torus torus;
	torus.objIdx = idx;
	torus.rc2 = a * a, torus.rt2 = b * b;
	torus.T = Identity();
	torus.invT = Identity();
	torus.r2 = sqrt( a + b );
	return torus;
}
void IntersectTorus( struct Torus *torus, struct Ray *ray )
{
	// via: https://www.shadertoy.com/view/4sBGDy
	float3 O = TransformPosition( ray->O, torus->invT );
	float3 D = TransformVector( ray->D, torus->invT );
	// extension rays need double precision for the quadratic solver!
	double po = 1, m = dot( O, O ), k3 = dot( O, D ), k32 = k3 * k3;
	// bounding sphere test
	double v = k32 - m + torus->r2;
	if (v < 0) return;
	// setup torus intersection
	double k = (m - torus->rt2 - torus->rc2) * 0.5, k2 = k32 + torus->rc2 * D.z * D.z + k;
	double k1 = k * k3 + torus->rc2 * O.z * D.z, k0 = k * k + torus->rc2 * O.z * O.z - torus->rc2 * torus->rt2;
	// solve quadratic equation
	if (fabs( k3 * (k32 - k2) + k1 ) < 0.0001)
	{
		swap( k1, k3 );
		po = -1, k0 = 1 / k0, k1 = k1 * k0, k2 = k2 * k0, k3 = k3 * k0, k32 = k3 * k3;
	}
	double c2 = 2 * k2 - 3 * k32, c1 = k3 * (k32 - k2) + k1;
	double c0 = k3 * (k3 * (-3 * k32 + 4 * k2) - 8 * k1) + 4 * k0;
	c2 *= 0.33333333333, c1 *= 2, c0 *= 0.33333333333;
	double Q = c2 * c2 + c0, R = 3 * c0 * c2 - c2 * c2 * c2 - c1 * c1;
	double h = R * R - Q * Q * Q, z;
	if (h < 0)
	{
		const double sQ = sqrt( Q );
		z = 2 * sQ * cos( acos( R / (sQ * Q) ) * 0.33333333333 );
	}
	else
	{
		const double sQ = cbrt( sqrt( h ) + fabs( R ) ); // pow( sqrt( h ) + fabs( R ), 0.3333333 );
		z = copysign( fabs( sQ + Q / sQ ), R );
	}
	z = c2 - z;
	double d1 = z - 3 * c2, d2 = z * z - 3 * c0;
	if (fabs( d1 ) < 1.0e-8)
	{
		if (d2 < 0) return;
		d2 = sqrt( d2 );
	}
	else
	{
		if (d1 < 0) return;
		d1 = sqrt( d1 * 0.5 ), d2 = c1 / d1;
	}
	double t = 1e20;
	h = d1 * d1 - z + d2;
	if (h > 0)
	{
		h = sqrt( h );
		double t1 = -d1 - h - k3, t2 = -d1 + h - k3;
		t1 = (po < 0) ? 2 / t1 : t1, t2 = (po < 0) ? 2 / t2 : t2;
		if (t1 > 0) t = t1;
		if (t2 > 0) t = min( t, t2 );
	}
	h = d1 * d1 - z - d2;
	if (h > 0)
	{
		h = sqrt( h );
		double t1 = d1 - h - k3, t2 = d1 + h - k3;
		t1 = (po < 0) ? 2 / t1 : t1, t2 = (po < 0) ? 2 / t2 : t2;
		if (t1 > 0) t = min( t, t1 );
		if (t2 > 0) t = min( t, t2 );
	}
	float ft = (float)t;
	if (ft > 0 && ft < ray->t) ray->t = ft, ray->objIdx = torus->objIdx;
}
bool IsOccludedTorus( struct Torus *torus, struct Ray *ray )
{
	// via: https://www.shadertoy.com/view/4sBGDy
	float3 O = TransformPosition( ray->O, torus->invT );
	float3 D = TransformVector( ray->D, torus->invT );
	float po = 1, m = dot( O, O ), k3 = dot( O, D ), k32 = k3 * k3;
	// bounding sphere test
	float v = k32 - m + torus->r2;
	if (v < 0.0) return false;
	// setup torus intersection
	float k = (m - torus->rt2 - torus->rc2) * 0.5f, k2 = k32 + torus->rc2 * D.z * D.z + k;
	float k1 = k * k3 + torus->rc2 * O.z * D.z, k0 = k * k + torus->rc2 * O.z * O.z - torus->rc2 * torus->rt2;
	// solve quadratic equation
	if (fabs( k3 * (k32 - k2) + k1 ) < 0.01f)
	{
		swap( k1, k3 );
		po = -1, k0 = 1 / k0, k1 = k1 * k0, k2 = k2 * k0, k3 = k3 * k0, k32 = k3 * k3;
	}
	float c2 = 2 * k2 - 3 * k32, c1 = k3 * (k32 - k2) + k1;
	float c0 = k3 * (k3 * (-3 * k32 + 4 * k2) - 8 * k1) + 4 * k0;
	c2 *= 0.33333333333f, c1 *= 2, c0 *= 0.33333333333f;
	float Q = c2 * c2 + c0, R = 3 * c0 * c2 - c2 * c2 * c2 - c1 * c1;
	float h = R * R - Q * Q * Q, z = 0;
	if (h < 0)
	{
		const float sQ = sqrt( Q );
		z = 2 * sQ * cos( acos( R / (sQ * Q) ) * 0.3333333f );
	}
	else
	{
		const float sQ = cbrtf( sqrt( h ) + fabs( R ) ); // powf( sqrt( h ) + fabs( R ), 0.3333333f );
		z = copysign( fabs( sQ + Q / sQ ), R );
	}
	z = c2 - z;
	float d1 = z - 3 * c2, d2 = z * z - 3 * c0;
	if (fabs( d1 ) < 1.0e-4f)
	{
		if (d2 < 0) return false;
		d2 = sqrt( d2 );
	}
	else
	{
		if (d1 < 0.0) return false;
		d1 = sqrt( d1 * 0.5f ), d2 = c1 / d1;
	}
	float t = 1e20f;
	h = d1 * d1 - z + d2;
	if (h > 0)
	{
		float t1 = -d1 - sqrt( h ) - k3;
		t1 = (po < 0) ? 2 / t1 : t1;
		if (t1 > 0 && t1 < ray->t) return true;
	}
	h = d1 * d1 - z - d2;
	if (h > 0)
	{
		float t1 = d1 - sqrt( h ) - k3;
		t1 = (po < 0) ? 2 / t1 : t1;
		if (t1 > 0 && t1 < ray->t) return true;
	}
	return false;
}
float3 GetNormalTorus( struct Torus *torus, float3 I )
{
	float3 L = TransformPosition( I, torus->invT );
	float3 temp;
	temp.x = 1, temp.y = 1, temp.z = -1;
	float3 N = normalize( L * (dot( L, L ) - torus->rt2 - torus->rc2 * temp) );
	return TransformVector( N, torus->T );
}
float3 GetAlbedo( struct Torus torus, float3 I )
{
	return 1 ; // material.albedo;
}

#ifdef FOURLIGHTS
	struct Quad quad[4];
#else
	struct Quad quad, dummyQuad1, dummyQuad2, dummyQuad3;
#endif
struct Sphere sphere;
struct Sphere sphere2;
struct Cube cube;
struct Plane plane[6];
struct Torus torus;

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

	struct mat4 identity = Identity();

	float3 pos;

		// we store all primitives in one continuous buffer
#ifdef FOURLIGHTS
	for (int i = 0; i < 4; i++) quad[i] = GetQuad( 0, 0.5f, identity );	// 0: four light sources
#else
	quad = Quad( 0, 1 );									// 0: light source
#endif
	sphere = GetSphere( 1, 0, 0.6f );				// 1: bouncing ball
	pos.x = 0, pos.y = 2.5f, pos.z = -3.07f; sphere2 = GetSphere( 2, pos, 8 );	// 2: rounded corners
	cube = GetCube( 3, 0, 1.15f, identity );			// 3: cube
	pos.x = 1, pos.y = 0,pos.z = 0; plane[0] = GetPlane( 4, pos, 3 );			// 4: left wall
	pos.x = -1, pos.y = 0, pos.z = 0; plane[1] = GetPlane( 5, pos, 2.99f );		// 5: right wall
	pos.x = 0, pos.y = 1, pos.z = 0; plane[2] = GetPlane( 6, pos, 1 );			// 6: floor
	pos.x = 0, pos.y = -1, pos.z = 0; plane[3] = GetPlane( 7, pos, 2 );			// 7: ceiling
	pos.x = 0, pos.y = 0, pos.z = 1; plane[4] = GetPlane( 8, pos, 3 );			// 8: front wall
	pos.x = 0, pos.y = 0, pos.z = -1; plane[5] = GetPlane( 9, pos, 3.99f );		// 9: back wall
	torus = GetTorus( 10, 0.8f, 0.25f );						// 10: torus
	torus.T = mat4Mul( Translate( -0.25f, 0, 2), RotateX( PI / 4 ) );
	torus.invT = Inverted( torus.T );

	//bool flag = IsOccludedSphere( &sphere, &ray);
}