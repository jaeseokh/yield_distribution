#' ---
#' title: "Quantile Forest Analysis"
#' author: "Taro Mieno"
#' output:
#'   html_document:
#'     number_sections: yes
#'     theme: flatly
#'     highlight: zenburn
#'     toc_float: yes
#'     toc: yes
#'     toc_depth: 3
#' geometry: margin=1in
#' ---

#+ setup, include=FALSE
knitr::opts_chunk$set(
  echo = TRUE,
  cache = FALSE,
  comment = NA,
  message = FALSE,
  warning = FALSE,
  tidy = FALSE,
  cache.lazy = FALSE,
  #--- figure ---#
  dpi=400,
  fig.width=7.5,
  fig.height=5,
  out.width="750px",
  out.height="500px"
)

# opts_knit$set(
#   root.dir = ""
# )

#/*=================================================*/
#' # Preparation
#/*=================================================*/
setwd('~/Dropbox/CollaborativeResearch/QuantileForest_PA')

library(sf)
library(grf)
library(data.table)
library(tidyverse)
library(magrittr)
library(parallel)

#/*=================================================*/
#' # Quantile Forest Analysis: unknown
#/*=================================================*/
#/*----------------------------------*/
#' ## Import data
#/*----------------------------------*/
#=== import the data ===#
# data_sf <- st_read('./Data/snider.gpkg')
data_sf <- st_read('./Data/finaldata_taro.gpkg') %>%
  na.omit() %>%
  rename(tgtn=NRATE_Gal3,tgts=SEEDRATE)

data_dt <- data.table(data_sf) %>%
  setnames(names(.),tolower(names(.)))
  # .[!is.na(slope),]

# data_dt[,.(ecs,tpi,twi,slope,om,elev,clay,silt,sand)] %>% cor
# lm(ecs~slope+elev+om+tpi+clay+silt+sand,data=data_dt) %>% summary()

#/*----------------------------------*/
#' ## Correlation matrix
#/*----------------------------------*/
var_ls_all <- c('n','seed','ecs','slope','dem','curv','tpi','twi','clay','silt','sand','water_storage')
data_dt[,var_ls_all,with=FALSE] %>% cor()

#/*----------------------------------*/
#' ## Simple viz
#/*----------------------------------*/
ggplot(data=data_dt) +
  geom_point(aes(y=yield,x=n))

ggplot(data=data_dt) +
  geom_point(aes(y=yield,x=seed))

#/*----------------------------------*/
#' ## Quantile Forest
#/*----------------------------------*/
#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### Define relevant variables
#/*~~~~~~~~~~~~~~~~~~~~~~*/
X <- data_dt[,var_ls_all,with=FALSE]
Y <- data_dt[,yield]

# q_seq <- seq(0.01,0.99,by=0.01)
q_seq <- seq(0.05,0.95,by=0.05)
qf <- quantile_forest(X,Y,quantiles=q_seq)

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### Prediction
#/*~~~~~~~~~~~~~~~~~~~~~~*/
n_max <- data_dt[,n] %>% max()
n_min <- data_dt[,n] %>% min()

N_seq <- seq(n_min,n_max,length=5)

X_test <- data_dt[100,var_ls_all,with=FALSE] %>%
  .[rep(1,length(N_seq)),] %>%
  .[,n:=N_seq]

pred_data <- predict(qf, X_test, quantiles=q_seq) %>%
  data.table() %>%
  setnames(names(.),paste(q_seq)) %>%
  .[,n:=N_seq] %>%
  melt(id.var='n') %>%
  .[,p:=as.numeric(as.character(variable))]

#=== distirbution ===#
library(mgcv)
library(gratia)
gam_res <- pred_data %>%
  split(.$n) %>%
  purrr::map(~ gam(p~s(value,k=8),data=.)) %>%
  purrr::map(~ fderiv(.,term='value')) %>%
  purrr::imap(., ~ data.table(yield=.x$eval,f=.x$derivatives[[1]]$deriv,N=as.numeric(.y))) %>%
  rbindlist()

ggplot(data=gam_res) +
  geom_line(aes(x=yield.value,y=f.V1,color=factor(N)))


ls(pdf_res[[1]])
pdf_res[[1]]$eval
pdf_res[[1]]$derivatives[[1]]$deriv

g_pdensity <- ggplot(data=.) +
    geom_line(aes(y=p,x=value,color=factor(n)))

ggsave(g_pdensity,file='./Results/quantile_forest_N.pdf',height=5,width=5)

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### Prediction
#/*~~~~~~~~~~~~~~~~~~~~~~*/
clay_max <- data_dt[,clay] %>% max()
clay_min <- data_dt[,clay] %>% min()

clay_seq <- seq(clay_min,clay_max,length=5)

X_test <- 
  data_dt[200,var_ls_all,with=FALSE] %>%
  .[rep(1,length(clay_seq)),] %>%
  .[,clay:=clay_seq]

predict(qf, X_test, quantiles=q_seq) %>%
  data.table() %>%
  setnames(names(.),paste(q_seq)) %>%
  .[,clay:=clay_seq] %>%
  melt(id.var='clay') %>%
  .[,p:=as.numeric(as.character(variable))] %>%
  ggplot(data=.) +
    geom_line(aes(y=p,x=value,color=factor(clay)))

#/*=================================================*/
#' # Quantile Forest Analysis: Snider
#/*=================================================*/
#/*----------------------------------*/
#' ## Import data
#/*----------------------------------*/
data_sf <- st_read('./Data/snider_high_res.gpkg') %>%
  na.omit() %>%
  rename(tgtn=NPlan,tgts=SRPlan)

data_dt <- data.table(data_sf) %>%
  setnames(names(.),tolower(names(.)))

#/*----------------------------------*/
#' ## Correlation matrix
#/*----------------------------------*/
var_ls_all <- c('nitrogen','seed','ecs','om','slope','elev','curv','tpi','twi','clay','silt','sand','water_storage')
data_dt[,var_ls_all,with=FALSE] %>% cor()

#/*----------------------------------*/
#' ## Quantile Forest
#/*----------------------------------*/
#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### Define relevant variables
#/*~~~~~~~~~~~~~~~~~~~~~~*/
X <- data_dt[,var_ls_all,with=FALSE]
Y <- data_dt[,yield]
q_seq <- seq(0.1,0.9,by=0.02)

qf <- quantile_forest(X,Y,quantiles=q_seq)

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### Prediction (seed)
#/*~~~~~~~~~~~~~~~~~~~~~~*/
seed_max <- data_dt[,seed] %>% max()
seed_min <- data_dt[,seed] %>% min()

S_seq <- seq(seed_min,seed_max,length=5)

X_test_S <- data_dt[200,var_ls_all,with=FALSE] %>%
  .[rep(1,length(S_seq)),] %>%
  .[,seed:=S_seq]

g_pdensity <- 
  predict(qf, X_test_S, quantiles=q_seq) %>%
  data.table() %>%
  setnames(names(.),paste(q_seq)) %>%
  .[,seed:=S_seq] %>%
  melt(id.var='seed') %>%
  .[,p:=as.numeric(as.character(variable))] %>%
  ggplot(data=.) +
    geom_line(aes(y=p,x=value,color=factor(seed)))

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### Prediction (nitrogen)
#/*~~~~~~~~~~~~~~~~~~~~~~*/
nitrogen_max <- data_dt[,nitrogen] %>% max()
nitrogen_min <- data_dt[,nitrogen] %>% min()

N_seq <- seq(nitrogen_min,nitrogen_max,length=5)

X_test_N <- data_dt[200,var_ls_all,with=FALSE] %>%
  .[rep(1,length(N_seq)),] %>%
  .[,nitrogen:=N_seq]

g_pdensity <- predict(qf, X_test_N, quantiles=q_seq) %>%
  data.table() %>%
  setnames(names(.),paste(q_seq)) %>%
  .[,nitrogen:=N_seq] %>%
  melt(id.var='nitrogen') %>%
  .[,p:=as.numeric(as.character(variable))] %>%
  ggplot(data=.) +
    geom_line(aes(y=p,x=value,color=factor(nitrogen)))

#/*~~~~~~~~~~~~~~~~~~~~~~*/
#' ### Prediction (ecs)
#/*~~~~~~~~~~~~~~~~~~~~~~*/
tpi_max <- data_dt[,tpi] %>% max()
tpi_min <- data_dt[,tpi] %>% min()

N_seq <- seq(tpi_min,tpi_max,length=5)

X_test <- data_dt[200,var_ls_all,with=FALSE] %>%
  .[rep(1,length(N_seq)),] %>%
  .[,tpi:=N_seq]

g_pdensity <- predict(qf, X_test, quantiles=q_seq) %>%
  data.table() %>%
  setnames(names(.),paste(q_seq)) %>%
  .[,tpi:=N_seq] %>%
  melt(id.var='tpi') %>%
  .[,p:=as.numeric(as.character(variable))] %>%
  ggplot(data=.) +
    geom_line(aes(y=p,x=value,color=factor(tpi)))

