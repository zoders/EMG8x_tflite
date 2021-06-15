
#
# Main component of TF Micro project 'hello_world'.
#



idf_component_register(SRCS "app_main.c" main_functions.cc model.cc output_handler.cc constants.cc
                    INCLUDE_DIRS ".")