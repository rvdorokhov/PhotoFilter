#import "@local/gost732-2017:0.4.2": *
#import "@local/bmstu:0.3.0": *

#show: приложение.with(буква: "А", содержание: [ Исходный текст программы ])

#let files = (
  "generate_kadis700k.m",
  "generate_kadis700k_night.m",
  "dataset.py",
  "dataset_linux_night.py",
  "ML_first_generation.py",
  "ML_second_generation.py"
)

#for file in files {
  листинг(raw(read("../code/" + file)))[ Содержимое файла #file ] 
}