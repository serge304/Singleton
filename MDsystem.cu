#include <stdio.h>
#include <cuda_runtime.h>
#include "math_lib.cuh"
#include "math_func.cuh"

#include "MDsystem.cuh"

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

//******************* MDcell methods ***************************//

__constant__ Cell dc_Cell;

MDcell* MDcell::m_pInstance = NULL;

MDcell* MDcell::Instance()
{
	static Cleanup cleanup;
	if (!m_pInstance)
		m_pInstance = new MDcell();
	return m_pInstance;
}

MDcell::Cleanup::~Cleanup()
{
	delete MDcell::m_pInstance;
	MDcell::m_pInstance = NULL;	
}

MDcell::MDcell() : MDinterface("Cell")
{
	List.push_back(MDdata(&h_Data.H, 1, "Cell tensor", HDF5interface::H5matrix9()));

	for (int igpu =  dev_ptr.size(); igpu--;)	
	{
		checkCudaErrors(cudaSetDevice(igpu)); 
		checkCudaErrors(cudaGetSymbolAddress((void**)&dev_ptr[igpu], dc_Cell));
	}
}

void MDcell::Set(double3 Box, int ngpu)
{
	m_ngpu = ngpu;
	m_set = true;
	Set(Box);
}

void MDcell::Set(matrix9 H, int ngpu)
{
	m_ngpu = ngpu;
	m_set = true;
	Set(H);
}

void MDcell::Set(double3 Box) 
{
	h_Data.V = Box.x*Box.y*Box.z;
	h_Data.rV = 1.0/h_Data.V;
	h_Data.Box = Box;
	h_Data.rBox = inverse(h_Data.Box);
	h_Data.H = make_diagonal(Box);
	h_Data.rH = inverse(h_Data.H);
	UpdateDevice();
}

void MDcell::Set(matrix9 H) 
{
	matrix9 Ht = transpose(H);
	h_Data.V = det(H);
	h_Data.rV = 1.0/h_Data.V;
	h_Data.H = H;
	h_Data.rH = inverse(H);
	h_Data.Box = make_double3(norm(Ht.X), norm(Ht.Y), norm(Ht.Z));
	h_Data.rBox = inverse(h_Data.Box);	
	UpdateDevice();
}

void MDcell::ScaleH(matrix9 S) {
	matrix9 H = MultiplyByElement(h_Data.H, S);
	Set(H);	
}

void MDcell::UpdateDevice()
{
	for (int igpu = m_ngpu; igpu--;)	
	{
		checkCudaErrors(cudaSetDevice(igpu)); 
		checkCudaErrors(cudaMemcpyToSymbol(dc_Cell, &h_Data, sizeof(Cell)));
	}
}

double3 MDcell::CellWidth() {
	double3 W;
	matrix9 Ht = transpose(h_Data.H);
	W.x = h_Data.V / norm(cross(Ht.Y, Ht.Z));
	W.y = h_Data.V / norm(cross(Ht.Z, Ht.X));
	W.z = h_Data.V / norm(cross(Ht.X, Ht.Y));
	return W;
}

double3 MDcell::CellAngles() {
	double3 W;
	matrix9 Ht = transpose(h_Data.H);
	W.x = angle(Ht.X, Ht.Y);
	W.y = angle(Ht.Y, Ht.Z);
	W.z = angle(Ht.X, Ht.Z);
	return W;
}

double MDcell::GetMinLength() {	
	return min(CellWidth()); 
}

double MDcell::GetMaxLength() {
	return max(CellWidth()); 
}

double3 MDcell::GetLengthRatio() {
	double l = GetMinLength();
	return h_Data.Box/l;
}

void MDcell::Print() {printf("Box size: %f %f %f\n\n", h_Data.Box.x, h_Data.Box.y, h_Data.Box.z);}
