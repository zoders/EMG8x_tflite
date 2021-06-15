/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_event.h"
#include "main_functions.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"
#include "freertos/event_groups.h"
#include "main_functions.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_log.h"
#include "esp_system.h"
#define _USE_MATH_DEFINES // for C++
#include <cmath>
static const char *TAG = "tensorfow lite" ;
// Globals, used for compatibility with Arduino-style sketches.
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  int input_length;
  int inference_count = 0;
  constexpr int kTensorArenaSize = 81000;
  uint8_t* tensor_arena;

 }// namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  tensor_arena = (uint8_t*) malloc(  kTensorArenaSize * sizeof(uint8_t) );
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  ESP_LOGI(TAG, "[APP] Free memory: %d bytes", esp_get_free_heap_size()) ;
  if (NULL == tensor_arena)
   TF_LITE_REPORT_ERROR(error_reporter, "err mem");
  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;
  resolver.AddExpandDims();
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  input = interpreter->input(0);
  input_length = input->bytes / sizeof(float);
  inference_count = 0;
}
void free_tensor_arena(){
  //free(tensor_arena);
}
// The name of this function is important for Arduino compatibility.
void predicate(float* signal, int muap_left, int muap_right, float* output) {
  for (int i = muap_left + 300; i < muap_right - 300; i++){
    input->data.f[i-muap_left] = signal[i];
  }
  interpreter->Invoke();
  float *output_vec;
  output_vec = interpreter->output(0)->data.f;
  HandleOutput(error_reporter, output_vec[0], output_vec[1], output_vec[2]);
  output[0] =  output_vec[0];
  output[1] =  output_vec[1];
  output[2] =  output_vec[2];
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
float float_abs(float f) {
    if (f < 0)
        return -f;
    return f;
}
float* muap(float* signal, int lenght) {
    float avg_abs = 0;
    float max = 0;
    for (int i = 0; i < lenght; i++){
      avg_abs += float_abs(signal[i]);
      if (signal[i] > max)
        max = signal[i];
    }
    avg_abs /= lenght;
    float t = 0;
    if (max > (30 * avg_abs))
      t = 5 * avg_abs;
    else
      t = max / 5;
    int wnd_size = 1200;
    int muap_left = 600;
    int muap_right = 600;
    int num_muaps = 0;
    float* muaps = new float[100];
    muaps[0] = 0;

    for (int k = 0; k < lenght; k++) {
        float* xWnd = new float[wnd_size];
        float mean = 0;
        float max_value = 0;
        int max_index = 0;
        for (int j = k; j < k + wnd_size; j++) {
            float f_a = float_abs(signal[j]);
            xWnd[j - k] = f_a;
            
        }
        
        mean /= (float)wnd_size;

        for (int j = k; j < k + wnd_size; j++) {
            xWnd[j - k] = xWnd[j - k] - mean;
            if (max_value < xWnd[j - k]) {
                max_value = xWnd[j - k];
                max_index = j;
            }
        }
        if ((max_index - k == wnd_size / 2) && (max_value > 15000)) {
            int a = k + wnd_size / 2 - muap_left;
            int b = k + wnd_size / 2 + muap_right;
            if ((a < 0 )|| (b > lenght)) continue;
            if (a < muaps[num_muaps*2])
              continue;
            muaps[num_muaps*2 + 1] = a; muaps[num_muaps*2 + 2] = b;
            num_muaps++;
            muaps[0] = num_muaps;
          vTaskDelay(10 / portTICK_PERIOD_MS) ;
        }
        delete[] xWnd;
    }
  return muaps;
}
void centrize(float* signal, int length){
  float sum_of_numbers = 0;
  for (int i = 192; i < length; i++){
    sum_of_numbers = sum_of_numbers + signal[i];
  }
  ESP_LOGI(TAG, "sum %f", sum_of_numbers) ;
  float mean = sum_of_numbers/((float)length-192);
  for (int i = 0; i < length; i++)
    signal[i] = signal[i] - mean;
}
void convolve (const float* in, float* out, int length)
{
  for (int i = 0; i < length; i++){
    out[i] = 0.;
    for (int j = 0; j < 513; j++)
      if(i - j >= 0)
        out[i] += float(hflt[j] * (double)in[i - j]);
    vTaskDelay(5 / portTICK_PERIOD_MS) ;
    if (i % (length / 100) == 0){
      if ((int)(i / (length / 100)) % 10 == 0)
        vTaskDelay(10 / portTICK_PERIOD_MS) ;
      ESP_LOGI("Convolve", "%d %%", (int)(i / (length / 100)) );
    }
  }
}
void except_empty_space(float* signal, bool* b, int length){
  for (int i = 192; i < length; i++){
    if (!b[i])
      signal[i] = signal[i - 1];
  }

}