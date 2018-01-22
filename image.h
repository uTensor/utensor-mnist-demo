#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include "utensor/tensor.hpp"

template<typename T, template<typename> typename TENSOR=RamTensor>
class Image {
	private:
		Tensor* data;
		int xDim;
		int yDim;
		bool dimOverride = false;
	public:

		Image(uint32_t x, uint32_t y){
			data = new TENSOR<T>();
			std::vector<uint32_t> tmp({x, y});
			data->init(tmp);
		}
		Image(Tensor* that): data(that){}
		Image(): data(nullptr){}
		
		T& operator[](int idx) { return *((T*)data->write<T>(idx, 0)); }
		// T& operator[](int idx) { return *write(idx, 0); }
		T& operator()(int x, int y){ return *((T*)data->write<T>(y*this->get_xDim() + x, 0)); }
		const T& operator[](int idx) const { return *data->read<T>(idx, 0); }
		const T& operator()(int x, int y) const{ return *data->read<T>(y*this->get_xDim() + x, 0); }

		Tensor* get_data() { return data; }
		void reshape(int x, int y){
			xDim = x;
			yDim = y;
			dimOverride = true;
		}
		int get_xDim(void) const {
			if(!dimOverride)
				return data->getShape()[0];
			else
				return xDim;
		}

		int get_yDim(void) const {
			if(!dimOverride)
				return data->getShape()[1];
			else 
				return yDim;
		}

		~Image(){
			delete data;
		}

};

template<typename T>
void get_bounding_box(const Image<T>& img, int& xMin, int& yMin, int& xMax, int& yMax){
	xMin = 5000000;
	yMin = 5000000;
	xMax = 0;
	yMax = 0;

	for(int i = 0; i < img.get_xDim(); i++){
		for(int j = 0; j < img.get_yDim(); j++){
			if(img(i,j) > 0){
				xMin = (i < xMin) ? i : xMin;
				yMin = (j < yMin) ? j : yMin;
				xMax = (i > xMax) ? i : xMax;
				yMax = (j > yMax) ? j : yMax;
			}
		}
	}
	// printf("%d, %d, %d, %d\n", xMin, yMin, xMax, yMax);
	return;
}

template<typename T>
void get_centroid(const Image<T>& img, int& xC, int& yC){

	xC = 0; 
	yC = 0;
	int xN = 0;
	int yN = 0;

	for(int i = 0; i < img.get_xDim(); i++){
		for(int j = 0; j < img.get_yDim(); j++){
			if(img(i,j) > 0){
				xC += i;
				yC += j;
				xN += 1;
				yN += 1;
			}
		}
	}
	xC /= xN;
	yC /= yN;
	return;
}

/**
 * @brief Chop an image 
 * @details Get the minimum bounding box of an image
 * 
 * @param img [description]
 * @tparam T Probably uint8_t
 * @return chopped image
 */
template<typename T> 
Image<T> chop(const Image<T>& img){
	int xMin, xMax, yMin, yMax;
	get_bounding_box(img, xMin, yMin, xMax, yMax);
	printf("Chopping image to bound = %d, %d, %d, %d\n", xMin, yMin, xMax, yMax);

	Image<T> temp(xMax-xMin, yMax-yMin);

	for(int i=0, ii=xMin; ii < xMax; i++, ii++){
		for(int j=0, jj=yMin; jj < yMax; j++, jj++){
			temp(i,j) = img(ii,jj);
		}
	}

	return temp;
}

/**
 * @brief Nearest interpolation
 * @details Stretch or shrink an image naively
 * 
 * @param img [description]
 * @param w2 new width
 * @param h2 new height
 * @tparam T [description]
 * @return [description]
 */
template<typename T>
Image<T> resize(const Image<T>& img, int w2, int h2){
    Image<T> temp(w2,h2);
    int x_ratio = (int)((img.get_xDim()<<16)/w2) +1;
    int y_ratio = (int)((img.get_yDim()<<16)/h2) +1;
    int x2, y2 ;

    for(int i = 0; i < h2; i++) {
        for(int j = 0; j < w2; j++) {
            x2 = ((j*x_ratio)>>16) ;
            y2 = ((i*y_ratio)>>16) ;
            temp[(i*w2) + j] = img[(y2*img.get_xDim()) + x2] ;
        }                
    }                
    return temp;

}


/**
 * @brief Zero Pad each side of the image
 *
 * @param img
 * @param padX number of pixels to pad the width with
 * @param padY number of pixels to pad the height with
 */

template<typename T>
Image<T> pad(const Image<T>& img, int padX, int padY){
	Image<T> temp(img.get_xDim() + 2*padX, img.get_yDim() + 2*padY);

	// Init to zero
	for(int i = 0; i < temp.get_xDim(); i++){
		for(int j = 0; j < temp.get_yDim(); j++){
			temp(i,j) = 0;
		}
	}
	for(int i = 0, ii = padX; i < img.get_xDim(); ii++, i++){
		for(int j = 0, jj = padY; j < img.get_yDim(); jj++, j++){
			temp(ii,jj) = img(i,j);
		}
	}

    return temp;

}

template<typename T>
void printImage(const Image<T>& img){
    cout << "{\n";
    for(int i = 0; i < img.get_xDim(); i++){
        cout << "[ ";
        for(int j = 0; j < img.get_yDim(); j++){
            cout << img(i,j) << ", ";
        }
        cout << "]" << std::endl;
    }
    cout << "}\n";

}
#endif
