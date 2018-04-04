#include "mbed.h"
#include "stm32f413h_discovery.h"
#include "stm32f413h_discovery_ts.h"
#include "stm32f413h_discovery_lcd.h"
#include "tensor.hpp"
#include "image.h"
#include "deep_mnist_mlp.hpp"
#include "models/deep_mlp.hpp"

#ifndef TARGET_SIMULATOR
#include "FATFileSystem.h"
#include "F413ZH_SD_BlockDevice.h"
Serial pc(USBTX, USBRX, 115200);
F413ZH_SD_BlockDevice bd;
FATFileSystem fs("fs");
#else
#define USER_BUTTON     BUTTON1
#endif

InterruptIn button(USER_BUTTON);

TS_StateTypeDef  TS_State = {0};

volatile bool trigger_inference = false;

void trigger_inference_cb(void){ trigger_inference = true; }

template<typename T>
void clear(Image<T>& img){
    for(int i = 0; i < img.get_xDim(); i++){
        for(int j = 0; j < img.get_yDim(); j++){
            img(i,j) = 0;
        }
    }
}

template<typename T>
void printImage(const Image<T>& img){

    for(int i = 0; i < img.get_xDim(); i++){
        for(int j = 0; j < img.get_yDim(); j++){
            printf("%f, ", img(i,j));
        }
        printf("]\n\r");
    }
}

/**
 * Simple box filter with extra weight on the center element.
 * Blurs the image to make it more realistic.
 */
template<typename T>
Image<T> box_blur(const Image<T>& img){
    Image<T> tmp(img.get_xDim(), img.get_yDim());
    clear(tmp);
    for(int i = 4; i < img.get_xDim() - 4; i++){
        for(int j = 4; j < img.get_yDim() - 4; j++){
            tmp(i,j) = img(i-1, j-1) + img(i, j-1) + img(i+1, j-1) +
                       img(i-1, j) + 3.0*img(i, j) + img(i+1, j) +
                       img(i-1, j+1) + img(i, j+1) + img(i+1, j+1);
            tmp(i,j) /= 11.0;
        }
    }

    return tmp;
}


int main()
{
    uint16_t x1, y1;
    printf("uTensor deep learning character recognition demo\n");
    printf("https://github.com/uTensor/utensor-mnist-demo\n");
    printf("Draw a number (0-9) on the touch screen, and press the button...\r\n");

#ifndef TARGET_SIMULATOR
    ON_ERR(bd.init(), "SDBlockDevice init ");
    ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");
#endif

    Image<float>* img = new Image<float>(240, 240);

    BSP_LCD_Init();
    button.rise(&trigger_inference_cb);

    /* Touchscreen initialization */
    if (BSP_TS_Init(BSP_LCD_GetXSize(), BSP_LCD_GetYSize()) == TS_ERROR) {
        printf("BSP_TS_Init error\n");
    }

    /* Clear the LCD */
    BSP_LCD_Clear(LCD_COLOR_WHITE);

    /* Set Touchscreen Demo1 description */
    //BSP_LCD_SetTextColor(LCD_COLOR_GREEN);
    //BSP_LCD_FillRect(0, 0, BSP_LCD_GetXSize(), 40);
    //BSP_LCD_SetTextColor(LCD_COLOR_BLACK);
    //BSP_LCD_SetBackColor(LCD_COLOR_GREEN);
    //BSP_LCD_SetFont(&Font16);
    //BSP_LCD_DisplayStringAt(0, 15, (uint8_t *)"Touch the screen", CENTER_MODE);

    Context ctx;
    clear(*img);


    while (1) {
        BSP_TS_GetState(&TS_State);
        if(trigger_inference){

            Image<float> smallImage = resize(*img, 64, 64);


            Image<float> chopped = chop(smallImage);
            printf("Done chopping\n\n");
            Image<float> img20   = resize(chopped, 20, 20);
            printf("Done resizing\n\n");
            Image<float> img28   = pad(img20, 4, 4);

            // Image processing is heavy on constrained devices
            // manually delete
            printf("Done padding\n\n");
            smallImage.~Image<float>();
            chopped.~Image<float>();
            img20.~Image<float>();
            delete img;

            Image<float> img28_2 = box_blur(img28);
            printf("Done blurring\n\r");
            img28.~Image<float>();

            printf("Reshaping\n\r");
            img28_2.get_data()->resize({1, 784});
            printf("Creating Graph\n\r");

            get_deep_mlp_ctx(ctx, img28_2.get_data());
            printf("Evaluating\n\r");
            ctx.eval();
            S_TENSOR prediction = ctx.get({"y_pred:0"});
            int result = *(prediction->read<int>(0,0));

            printf("Number guessed %d\n\r", result);

            BSP_LCD_Clear(LCD_COLOR_WHITE);
            BSP_LCD_SetTextColor(LCD_COLOR_BLACK);
            BSP_LCD_SetFont(&Font24);
            uint8_t number[2];
            number[1] = 0;
            number[0] = 48 + result;
            BSP_LCD_DisplayStringAt(0, 120, number, CENTER_MODE);
            trigger_inference = false;
            img = new Image<float>(240, 240);
            clear(*img);
            break;
        }
        if(TS_State.touchDetected) {
            /* One or dual touch have been detected          */

            /* Get X and Y position of the first touch post calibrated */
            x1 = TS_State.touchX[0];
            y1 = TS_State.touchY[0];

            img->draw_circle(x1, y1, 7); //Screen not in image x,y format. Must transpose

            BSP_LCD_SetTextColor(LCD_COLOR_GREEN);
            BSP_LCD_FillCircle(x1, y1, 5);

            wait_ms(5);
        }
    }
}
