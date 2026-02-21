#import "@local/gost732-2017:0.4.2": *
#import "@local/bmstu:0.3.0": *

#show: гост732-2017

// #страница(image("материалы/титул_финал.jpg", height: 100%), номер: нет)

// #страница(image("материалы/приватно/ТРПС задание на курсовую работы.jpg", height: 100%), номер: нет)

#содержание()

#include "разделы/1-введение.typ"

#include "разделы/2-анализ-требований.typ"

#include "разделы/4-разработка-нейронной-сети.typ"

#include "разделы/9-заключение.typ"

#set bibliography(style: "bib.csl")
#show bibliography: it_bib => {
  set block(inset: 0pt)
  show block: it_block => {
    par(it_block.body)
  }
  it_bib
}
#bibliography("bibliography.yml")

#include "разделы/прил-д-код.typ"