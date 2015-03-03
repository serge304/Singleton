#ifndef MDSYSTEMCUH
#define MDSYSTEMCUH

#include <vector>
#include <stdexcept>
#include <H5Cpp.h>

#include "GPUutils.cuh"
#include "math_types.cuh"
#include "io.cuh"

// Singleton base template class

template<typename T>
class MDbase
{
protected:
	T h_Data ;
	std::vector<T*> dev_ptr;

	int m_ngpu;
	bool m_set;

	MDbase() { 
		m_set = false; 
		int deviceCount = 0;
		checkCudaErrors(cudaGetDeviceCount(&deviceCount));
		dev_ptr.resize(deviceCount);	
	}
public:
	const T& GetHostReference()	{
		if (!m_set) throw std::runtime_error("ERROR: instance is called before setup");
		return h_Data;	
	}
	std::vector<T*> GetDevicePointers() {
		if (!m_set) throw std::runtime_error("ERROR: instance is called before setup");
		return dev_ptr;
	}
	void Set(T &data) {h_Data = data;}
};


//******************* MDcell ***************************//

struct Cell {
	double V, rV;
	double3 Box, rBox;
	matrix9 H, rH;
};

class MDcell : public MDbase<Cell>, public MDinterface
{
private:
	static MDcell *m_pInstance;  // pointer to class instance		            
	friend class Cleanup;
	class Cleanup
	{
	public:
		~Cleanup();
	};

	void UpdateDevice();
	MDcell();                         // private constructor

public:
	static MDcell* Instance();
	void Set(double3, int);
	void Set(matrix9, int);
	void Set(double3);
	void Set(matrix9 H);
	void ScaleH(matrix9 S);
	double3 GetBox() {return h_Data.Box;}
	double3 GetrBox() {return h_Data.rBox;}
	matrix9 GetH() {return h_Data.H;}
	matrix9 GetrH() {return h_Data.rH;}
	double GetV() {return h_Data.V;}
	double GetrV() {return h_Data.rV;}
	double3 CellAngles();
	double3 CellWidth();
	double GetMinLength();
	double GetMaxLength();
	double3 GetLengthRatio();
	void Print();
};



