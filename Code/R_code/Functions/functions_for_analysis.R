
# Function to generate all possible GAM formulas
#  for a given set of input variables 
#  and field regression variables
#  each input and set of field variables can be either linear or spline
#  for the spline function, input variables can choose k=3 or k=4
#  in case of field variables, all of them will be either linear or spline with k=3
#  the function returns a list of formulas


# Function to combine densities per group and N
make_density_dt_grouped <- function(tn) {
  mb <- mbme_density_all_models[abs(n - tn) < tol][, method := "MBME"]
  no <- gam_norm_densities[abs(n - tn) < tol][, method := "Normal"]
  qf_sub <- qf_result_dt[abs(n - tn) < tol]

  qf_list <- lapply(seq_len(nrow(qf_sub)), function(i) {
    row <- qf_sub[i]
    qv <- unlist(row[, ..quant_cols])
    if (length(unique(qv)) < 2 || any(is.na(qv))) return(NULL)
    d <- density(qv, from = 0, to = 1, n = 200)
    data.table(
      yield = d$x, density = d$y,
      prcp_t = row$prcp_t,
      group_label = row$group_label,
      method = "Quantile Forest"
    )
  })

  rbindlist(list(
    mb[, .(yield, density, prcp_t, group_label, method)],
    no[, .(yield, density, prcp_t, group_label, method)],
    rbindlist(qf_list, fill = TRUE)
  ), fill = TRUE)
}


estimate_mbme_density <- function(mu1, mu2, mu3, support = c(0, 1), n_grid = 500) {
  y_seq <- seq(support[1], support[2], length.out = n_grid)

  safe_exp <- function(x) {
    x <- pmin(pmax(x, -700), 700)
    exp(x)
  }

  moment_loss <- function(lambda) {
    exp_term <- safe_exp(lambda[1]*y_seq + lambda[2]*y_seq^2 + lambda[3]*y_seq^3)
    Z <- sum(exp_term) * (y_seq[2] - y_seq[1])
    if (!is.finite(Z) || Z <= 0) return(1e10)
    m1 <- sum(y_seq * exp_term) * (y_seq[2] - y_seq[1]) / Z
    m2 <- sum(y_seq^2 * exp_term) * (y_seq[2] - y_seq[1]) / Z
    m3 <- sum(y_seq^3 * exp_term) * (y_seq[2] - y_seq[1]) / Z
    sum((c(m1, m2, m3) - c(mu1, mu2, mu3))^2)
  }

  opt <- tryCatch(
    optim(par = c(1, -1, 0), fn = moment_loss, method = "BFGS"),
    error = function(e) NULL
  )

  if (is.null(opt) || opt$convergence != 0 || !is.finite(opt$value)) {
    for (i in 1:5) {
      init <- runif(3, -2, 2)
      opt_try <- tryCatch(
        optim(par = init, fn = moment_loss, method = "BFGS"),
        error = function(e) NULL
      )
      if (!is.null(opt_try) && opt_try$convergence == 0 && is.finite(opt_try$value)) {
        opt <- opt_try
        break
      }
    }
  }

  if (is.null(opt)) return(NULL)

  lambda <- opt$par
  exp_term <- safe_exp(lambda[1]*y_seq + lambda[2]*y_seq^2 + lambda[3]*y_seq^3)
  Z <- sum(exp_term) * (y_seq[2] - y_seq[1])
  density <- exp_term / Z

  return(data.table(yield = y_seq, density = density))
}



generate_gam_formulas <- function(input_vars, field_reg_vars) {
  # Define possible spline options for input_vars
  input_var_options <- list(
    "linear" = input_vars,
    "spline_k3" = paste0("s(", input_vars, ",k=3)"),
    "spline_k4" = paste0("s(", input_vars, ",k=4)")
  )
  
  # Possible choices for field_reg_vars (either all linear or all splines with k=3)
  field_reg_linear <- paste(field_reg_vars, collapse = "+")
  field_reg_spline <- paste0("s(", field_reg_vars, ",k=3)", collapse = "+")
  
  field_var_options <- list("linear" = field_reg_linear, "spline_k3" = field_reg_spline)
  
  # Function to generate formula for given choices of input and field variables
  create_formula <- function(input_var_comb, field_var_choice) {
    input_part <- paste(input_var_comb, collapse = "+")
    formula <- paste("yield ~", input_part, "+", field_var_choice)
    return(formula)
  }
  
  # Generate all combinations of input_vars and field_reg_vars
  formulas <- list()
  
  # If input_vars is of length 1
  if (length(input_vars) == 1) {
    input_combinations <- expand.grid(
      input_1 = names(input_var_options),
      field = names(field_var_options),
      stringsAsFactors = FALSE
    )
    
    # Generate formula for each combination
    for (i in seq_len(nrow(input_combinations))) {
      input_1_choice <- input_var_options[[input_combinations$input_1[i]]]
      field_choice <- field_var_options[[input_combinations$field[i]]]
      formulas[[i]] <- create_formula(input_1_choice, field_choice)
    }
  }
  
  # If input_vars is of length 2
  if (length(input_vars) == 2) {
    input_combinations <- expand.grid(
      input_1 = names(input_var_options),
      input_2 = names(input_var_options),
      field = names(field_var_options),
      stringsAsFactors = FALSE
    )
    
    # Generate formula for each combination
    for (i in seq_len(nrow(input_combinations))) {
      input_1_choice <- input_var_options[[input_combinations$input_1[i]]][1]
      input_2_choice <- input_var_options[[input_combinations$input_2[i]]][2]
      field_choice <- field_var_options[[input_combinations$field[i]]]
      formulas[[i]] <- create_formula(c(input_1_choice, input_2_choice), field_choice)
    }
  }
  
  return(formulas)
}


# Function to run and evaluate GAM models for a given set of formulas and a data set.
# The function fits GAM models using REML and GCV.Cp methods and by each formula,
# evaluates them based on REML and GCV scores.
# The function returns the two best models based on REML and GCV scores,respectively.


run_and_evaluate_gams <- function(gam_formulas, data) {
  results <- data.frame()
  
  # Loop through each formula
  for (i in seq_along(gam_formulas)) {
    formula <- as.formula(gam_formulas[[i]])
    
    # Fit the GAM model
    gam_model_gcv <- gam(formula, data = data, method = "GCV.Cp")
    
    # Extract evaluation metrics
    gcv_score <- gam_model_gcv$gcv.ubre  # GCV score
    
    # Store the results in a data frame
    results <- rbind(results, data.frame(
      formula = paste(gam_formulas[[i]]), 
      gcv = gcv_score
    ))

    rownames(results) <- NULL
  }

  best_gcv <- as.formula(results[which.min(results$gcv),"formula"])
  
    gam_gcv <- gam(best_gcv, data = data, method = "GCV.Cp")
    

 return(list(gam_best = gam_gcv))
}


     predict_yield_range <- function( formula,data_for_evaluation) {

  #--- predict yield ---#
  yield_prediction <- predict(formula, newdata =  data_for_evaluation, se = TRUE)

 evaluation_data <- data_for_evaluation[, .(s_rate)][, `:=`(
    yield_hat = yield_prediction$fit,
    yield_hat_se = yield_prediction$se.fit
     )]

  return(evaluation_data)

}

  estimate_profit_fcn <- function(info_table, evaluation_data, price_table) {
  # Get prices for the given year from price_tab
  corn_price_year <- price_table[Year == info_table$year, corn_price]
  seed_price_year <- price_table[Year == info_table$year, seed_price]

  # Calculate profit estimates in eval_data
  evaluation_table <- evaluation_data %>%
    .[, profit_hat_year := corn_price_year * yield_hat - seed_price_year * s_rate] %>%
    .[, profit_hat_low := corn_price_low * yield_hat - seed_price_low * s_rate] %>%
    .[, profit_hat_high := corn_price_high * yield_hat - seed_price_high * s_rate] %>%
    .[, profit_hat_se := corn_price_year * yield_hat_se]

  return(evaluation_table)
}




