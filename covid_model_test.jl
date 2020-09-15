push!(LOAD_PATH,pwd())
using DifferentialEquations
using StaticArrays
using StatsPlots; gr()
using LinearAlgebra
using Distributions
using Random
using Statistics
using DelimitedFiles
using MCMCChains

# define model and random walk functions
function delayed_ode(u,p,t)
  rates, probabilities, transmission_rates = p
  du1  = -u[1] * transmission_rates[1] * ( transmission_rates[2]*(u[2]+u[3]+u[4]+u[13]+u[14]+u[15]+u[16]) + (u[5]+u[6]+u[7]) ) / u[20]
  du2  = (1-probabilities[1]) * u[1] * transmission_rates[1] * ( transmission_rates[2]*(u[2]+u[3]+u[4]+u[13]+u[14]+u[15]+u[16]) + (u[5]+u[6]+u[7]) ) / u[20] - rates[1]*u[2]
  du3  = rates[1]*u[2] - rates[1]*u[3]
  du4  = rates[1]*u[3] - rates[1]*u[4]
  du5  = probabilities[2]*rates[1]*u[4] - rates[2]*u[5]
  du6  = rates[2]*u[5] - rates[2]*u[6]
  du7  = (1-probabilities[2])*rates[1]*u[4] - rates[3]*u[7]
  du8  = probabilities[3]*rates[2]*u[6] - rates[4]*u[8]
  du9  = (1-probabilities[3]-probabilities[5])*rates[2]*u[6] - rates[5]*u[9]
  du10 = probabilities[4]*rates[4]*u[8] - rates[6]*u[10]
  du11 = (1-probabilities[4])*rates[4]*u[8] - rates[7]*u[11]
  du12 = rates[7]*u[11] - rates[8]*u[12]
  du13 = probabilities[1] * u[1] * transmission_rates[1] * ( transmission_rates[2]*(u[2]+u[3]+u[4]+u[13]+u[14]+u[15]+u[16]) + (u[5]+u[6]+u[7]) ) / u[20] - rates[1]*u[13]
  du14 = rates[1]*u[13] - rates[1]*u[14]
  du15 = rates[1]*u[14] - rates[1]*u[15]
  du16 = rates[1]*u[15] - rates[9]*u[16]
  du17 = rates[3]*u[7] + rates[5]*u[9] + rates[8]*u[12] + rates[9]*u[16]
  du18 = rates[6]*u[10] + rates[11]*u[21] - rates[10]*u[18]
  du19 = rates[10]*u[18]
  du20 = -rates[6]*u[10] - rates[11]*u[21]
  du21 = probabilities[5]*rates[2]*u[6] - rates[11]*u[21]
  @SVector [du1,du2,du3,du4,du5,du6,du7,du8,du9,du10,du11,
   du12,du13,du14,du15,du16,du17,du18,du19,du20,du21]
end

function solve_ode(position,prediction_length::Int64)
  rates = copy(initial_rates);
  transmission_rates = copy(initial_transmission_rates);
  probabilities = copy(initial_probabilities);
  control_dates = copy(initial_control_dates);
  control_dates[end] = prediction_length;
  number_of_states = length(initial_state);
  rCD = rates[6];
  rHD = rates[11];
  beta = copy(transmission_rates[1]);
  reduced_beta = position[1:length(control_dates)-1]*beta;
  number_of_breakpoints = length(reduced_beta)
  log_initial_infectious = position[number_of_breakpoints+1];
  sigma_hi = position[number_of_breakpoints+2];
  sigma_hp = position[number_of_breakpoints+3];
  sigma_up = position[number_of_breakpoints+4];
  sigma_di = position[number_of_breakpoints+5];
  rCM = position[number_of_breakpoints+6];
  rHR = position[number_of_breakpoints+7];
  pC = position[number_of_breakpoints+8];
  pT = position[number_of_breakpoints+9];
  rates[[5,7]] = [rHR,rCM];
  probabilities[[3,5]] = [pC,pT];

  # solve the ODE
  Yt = Array{Float64,2}(undef,number_of_states,length(control_dates)+1)
  Y0 = [initial_population-exp(log_initial_infectious),(1-probabilities[1])*exp(log_initial_infectious),
        0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,probabilities[1]*exp(log_initial_infectious),0.,0.,0.,0.,0.,0.,initial_population,0.]
  Yt[:,1] = Y0;
  Yall = Array{Float64,2}(undef,number_of_states,Int(control_dates[end]+1))
  time_range = (0.,control_dates[1])
  t_eval = LinRange(0,control_dates[1],Int(control_dates[1])+1)
  params = [rates,probabilities,transmission_rates]
  prob = ODEProblem(delayed_ode,Y0,time_range,params)
  sol = solve(prob,saveat=t_eval)

  Yall[:,1:Int(control_dates[1])+1] = reduce(hcat, sol.u)
  Yt[:,2] = Yall[:,Int(control_dates[1]+1)]

  for ic in 1:length(control_dates)-1
      time_range = (control_dates[ic],control_dates[ic+1]+1)
      t_eval = LinRange(control_dates[ic],
                        control_dates[ic+1]+1,
                        Int(control_dates[ic+1]-control_dates[ic])+2)
      transmission_rates[1] = reduced_beta[ic]
      prob = ODEProblem(delayed_ode,
                        Yt[:,ic+1],
                        time_range,
                        params)
      sol = solve(prob,saveat=t_eval);
      Yall[:,Int(control_dates[ic]):Int(control_dates[ic+1])] = reduce(hcat, sol.u)[:,2:end]
      Yt[:,ic+2] = Yall[:,Int(control_dates[ic+1]-1)]
    end
  return Yall
end

function log_likelihood(position::Array{Float64})
  number_of_parameters = length(position);
  number_of_breakpoints = length(initial_control_dates)-1;
  if (any(position[[1,2,3,4,number_of_breakpoints+6,
                            number_of_breakpoints+7,
                            number_of_breakpoints+8,
                            number_of_breakpoints+9]] .< 0.0) ||
      any(position[[number_of_breakpoints+2,
                    number_of_breakpoints+3,
                    number_of_breakpoints+4,
                    number_of_breakpoints+5]] .< 1.0) ||
      any(position[[number_of_breakpoints+6,
                    number_of_breakpoints+7,
                    number_of_breakpoints+8,
                    number_of_breakpoints+9]] .>= 1.0) )
      return -Inf
  else
    rates = copy(initial_rates);
    transmission_rates = copy(initial_transmission_rates);
    probabilities = copy(initial_probabilities);
    control_dates = copy(initial_control_dates);
    number_of_states = length(initial_state);
    rCD = rates[6];
    beta = copy(transmission_rates[1]);
    reduced_beta = position[1:length(control_dates)-1]*beta;
    log_initial_infectious = position[number_of_breakpoints+1];
    sigma_hi = position[number_of_breakpoints+2];
    sigma_hp = position[number_of_breakpoints+3];
    sigma_up = position[number_of_breakpoints+4];
    sigma_di = position[number_of_breakpoints+5];
    rCM = position[number_of_breakpoints+6];
    rHR = position[number_of_breakpoints+7];
    rHD = rates[11];
    pC = position[number_of_breakpoints+8];
    pT = position[number_of_breakpoints+9];
    rates[[5,7]] = [rHR,rCM];
    probabilities[[3,5]] = [pC,pT];

    # solve the ODE
    Yt = Array{Float64,2}(undef,number_of_states,length(control_dates)+1)
    Y0 = [initial_population-exp(log_initial_infectious),(1-probabilities[1])*exp(log_initial_infectious),
          0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,probabilities[1]*exp(log_initial_infectious),0.,0.,0.,0.,0.,0.,initial_population,0.]
    Yt[:,1] = Y0;
    Yall = Array{Float64,2}(undef,number_of_states,Int(control_dates[end]+1))
    time_range = (0.,control_dates[1])
    t_eval = LinRange(0,control_dates[1],Int(control_dates[1])+1)
    params = [rates,probabilities,transmission_rates]
    prob = ODEProblem(delayed_ode,Y0,time_range,params)
    sol = solve(prob,saveat=t_eval)

    Yall[:,1:Int(control_dates[1])+1] = reduce(hcat, sol.u)
    Yt[:,2] = Yall[:,Int(control_dates[1]+1)]

    for ic in 1:length(control_dates)-1
        time_range = (control_dates[ic],control_dates[ic+1]+1)
        t_eval = LinRange(control_dates[ic],
                          control_dates[ic+1]+1,
                          Int(control_dates[ic+1]-control_dates[ic])+2)
        transmission_rates[1] = reduced_beta[ic]
        prob = ODEProblem(delayed_ode,
                          Yt[:,ic+1],
                          time_range,
                          params)
        sol = solve(prob,saveat=t_eval);
        Yall[:,Int(control_dates[ic]):Int(control_dates[ic+1])] = reduce(hcat, sol.u)[:,2:end]
        Yt[:,ic+2] = Yall[:,Int(control_dates[ic+1]-1)]
      end
    # Calculate log likelihood given fitting specification
    log_likelihood = 0.0
    if "hospital_incidence" in fit
      for (index,data) in enumerate(hospital_incidence_data)
        log_likelihood += logpdf(NegativeBinomial(rates[2]*Yall[6,hospital_incidence_indices[index]]/(sigma_hi-1),
                                                  1/sigma_hi),
                                 data)
      end
    end

    if "hospital_prevalence" in fit
      for (index,data) in enumerate(hospital_prevalence_data)
        log_likelihood += logpdf(NegativeBinomial(sum(Yall[[8,9,10,11,12,21],hospital_prevalence_indices[index]])/(sigma_hp-1),
                                                  1/sigma_hp),
                                 data)
      end
    end

    if "icu_prevalence" in fit
      for (index,data) in enumerate(icu_prevalence_data)
        log_likelihood += logpdf(NegativeBinomial(sum(Yall[[10,11],icu_prevalence_indices[index]])/(sigma_up-1),
                                                  1/sigma_up),
                                 data)
      end
    end

    if "death_incidence" in fit
      for (index,data) in enumerate(death_data)
        log_likelihood += logpdf(NegativeBinomial((rHD*Yall[21,death_indices[index]] + rCD*Yall[10,death_indices[index]])/(sigma_di-1),
                                                   1/sigma_di),
                                 data)
      end
    end
    # beta(7,7) prior on pD
    log_likelihood += 6*log(pD) + 6*log(1-pD)
    return log_likelihood
  end
end

function random_walk(model::Function,
                     number_of_samples::Int64,
                     initial_position::Array{Float64},
                     step_size::Float64,
                     proposal_covariance=I,
                     thinning_rate::Int64=1)

  println("Running RWM for $number_of_samples samples")
  # initialise the covariance proposal matrix
  number_of_parameters = length(initial_position)

  # check if default value is used, and set to q x q identity
  if isequal(proposal_covariance,I)
    identity = true
  else
    identity = false
    proposal_cholesky = cholesky(proposal_covariance).L
  end


  # initialise samples matrix and acceptance ratio counter
  accepted_moves = 0
  mcmc_samples = Array{Float64,2}(undef, number_of_samples, number_of_parameters)
  mcmc_samples[1,:] = initial_position
  number_of_iterations = number_of_samples*thinning_rate

  # initial markov chain
  current_position = initial_position
  current_log_likelihood = log_likelihood(current_position)

  for iteration_index = 1:number_of_iterations
    if iteration_index%(number_of_iterations/10)==0
      println("Progress: ",100*iteration_index/number_of_iterations,"%");
    end

    if identity
      proposal = current_position + step_size*rand(Normal(),number_of_parameters)
    else
      proposal = current_position + step_size*proposal_cholesky*reshape(rand(Normal(),number_of_parameters),(number_of_parameters,1))
    end
    proposal_log_likelihood = log_likelihood(proposal)
    if proposal_log_likelihood == -Inf
      if iteration_index%thinning_rate == 0
        mcmc_samples[Int(iteration_index/thinning_rate),:] = current_position
        iteration_index%thinning_rate == 0 && continue
      end
    end

    # accept-reject step
    if rand(Uniform()) < exp(proposal_log_likelihood - current_log_likelihood)
      current_position = proposal
      current_log_likelihood = proposal_log_likelihood
      accepted_moves += 1
    end
    if iteration_index%thinning_rate == 0
      mcmc_samples[Int(iteration_index/thinning_rate),:] = current_position
    end
  end # for loop
  println("Acceptance ratio:",accepted_moves/number_of_iterations)
  return mcmc_samples
end # function

# define initial conditions and parameters
# set rates
rE1 = 1.0/5.5;
rE = 3.0/5.5;
rIH1 = 1.0/5.0
rIH = 2.0/5.0;
rIR = 1.0/3.5;
rHC = 1.0/1.93; # 1.0/2.09;
rHR = 1.0/8.04;# 1.0/7.98;
rCD = 1.0/11.85; #10.85; # 1.0/8.23;
rMR = 1.0/6.15; # 1.0/5.93;
rCM = 1.0/16.5; #10.5; # 1.0/7.92;
rA = 1.0/3.5;
rX = 1.0/4.5;
rHD = 1/8.01; # 1.0/6.83;
global initial_rates = [rE,rIH,rIR,rHC,rHR,rCD,rCM,rMR,rA,rX,rHD];

# set proportions
pA = 0.18;
pH = 0.05; # 0.15
pC = 0.3; # 0.1755
pD = 0.48; # 0.3;
pU = 0.3;
pT = 0.23; # (1-pU)/pU * pC/(1-pC) * pD;
global initial_probabilities = [pA,pH,pC,pD,pT];
# set transmission parameters
R0 = 3.62;
f = 0.25;
k = (1-pA)*(f/rE1 + pH/rIH1 + (1-pH)/rIR) + pA*f*(1/rE1 + 1/rA); # (f/rE + pH/rIH + (1-pH)/rIR) + pA*f*(1/rE + 1/rA)
b = R0/k; # k = 4.34
global initial_transmission_rates = [b,f];
# choose region and fitting criteria
global regions = ["WA","SC","NI"];
for region in regions
  global fit = ["hospital_incidence",
                "hospital_prevalence",
                "icu_prevalence",
                "death_incidence"]; # hospital_incidence, hospital_prevalence, icu_prevalence, death_incidence

  println("Fitting $fit to $region")

  if region == "EE"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,1:4]
    # data = readdlm("datafit_EN.csv",',')[:,1:3];
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35;
    first_day_hp = 35;
    first_day_up = 35;
    first_death = 35;
    final_day = first_day_hp + length(hospital_prevalence_data);
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    global initial_population = 6200000.0;
    global region_gl = region;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  elseif region == "LO"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,5:8]
    # data = readdlm("datafit_EN.csv",',')[:,1:3];
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35;
    first_day_hp = 35;
    first_day_up = 35;
    first_death = 35;
    final_day = first_day_hp + length(hospital_prevalence_data);
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    global initial_population = 8900000.0;
    global region_gl = region;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  elseif region == "MI"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,9:12]
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35;
    first_day_hp = 35;
    first_day_up = 35;
    first_death = 35;
    final_day = first_day_hp + length(hospital_prevalence_data);
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    global initial_population = 10700000.0;
    global region_gl = region;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  elseif region == "NE"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,13:16];
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35;
    first_day_hp = 35;
    first_day_up = 35;
    first_death = 35;
    final_day = first_day_hp + length(hospital_prevalence_data);
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    global initial_population = 3210000.0;
    global region_gl = region;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  elseif region == "NW"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,17:20]
    # data = readdlm("datafit_EN.csv",',')[:,1:3];
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35;
    first_day_hp = 35;
    first_day_up = 35;
    first_death = 35;
    final_day = first_day_hp + length(hospital_prevalence_data);
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    global initial_population = 7300000.0;
    global region_gl = region;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  elseif region == "SE"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,21:24]
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35;
    first_day_hp = 35;
    first_day_up = 35;
    first_death = 35;
    final_day = first_day_hp + length(hospital_prevalence_data);
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    global initial_population = 9130000.0;
    global region_gl = region;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  elseif region == "SW"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,25:28]
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35;
    first_day_hp = 35;
    first_day_up = 35;
    first_death = 35;
    final_day = first_day_hp + length(hospital_prevalence_data);
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    global initial_population = 5600000.0;
    global region_gl = region;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  elseif region == "EN"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,29:32]
    # data = readdlm("datafit_EN.csv",',')[:,1:3];
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35;
    first_day_hp = 35;
    first_day_up = 35;
    first_death = 35;
    final_day = first_day_hp + length(hospital_prevalence_data);
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    global initial_population = 56000000.0;
    global region_gl = region;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  elseif region == "SC"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,33:36]
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35; # 39
    first_day_hp = 60;
    first_day_up = 61;
    first_death = 35; # 39
    final_day = first_day_hp + length(hospital_prevalence_data);
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    global initial_population = 5450000.0;
    global region_gl = region;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  elseif region == "WA"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,37:40]
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35; # 39
    first_day_hp = 121;
    first_day_up = 57;
    first_death = 35; # 39
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    # deal with NaN values midway through data for wales
    # hospital_incidence
    for (index,value) in enumerate(hospital_incidence_indices)
      if value > 139
        hospital_incidence_indices[index] += 2
      end
    end
    for (index,value) in enumerate(hospital_incidence_indices)
      if value > 146
        hospital_incidence_indices[index] += 2
      end
    end

    # hospital_prevalence
    for (index,value) in enumerate(hospital_prevalence_indices)
      if value > 139
      hospital_prevalence_indices[index] += 2
      end
    end
    for (index,value) in enumerate(hospital_prevalence_indices)
      if value > 146
      hospital_prevalence_indices[index] += 2
      end
    end

    # icu_prevalence
    for (index,value) in enumerate(icu_prevalence_indices)
      if value > 144
      icu_prevalence_indices[index] += 4
      end
    end

    # deaths
    for (index,value) in enumerate(death_indices)
      if value > 144
      death_indices[index] += 4
      end
    end

    global initial_population = 3140000.0;
    global region_gl = region;
    final_day = hospital_prevalence_indices[end]+2;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  elseif region == "NI"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,41:44]
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35;
    first_day_hp = 35;
    first_day_up = 57;
    first_death = 35;
    final_day = first_day_hp + length(hospital_prevalence_data);
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    global initial_population = 1880000.0;
    global region_gl = region;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  elseif region == "UK"
    # set data
    data = readdlm("all_data_to_export.csv",',')[:,45:48]
    global hospital_incidence_data  = [data[i,1] for i in 1:length(data[:,1]) if isnan.(data[i,1])==0][1:end-2]; #data[:,1];
    global hospital_prevalence_data = [data[i,2] for i in 1:length(data[:,2]) if isnan.(data[i,2])==0];# data[:,2];
    global icu_prevalence_data      = [data[i,3] for i in 1:length(data[:,3]) if isnan.(data[i,3])==0];# data[:,3];
    global death_data               = [data[i,4] for i in 1:length(data[:,4]) if isnan.(data[i,4])==0][1:end-5];# data[:,4];
    # construct indices for log likelihood
    first_day_hi = 35;
    first_day_hp = 60;
    first_day_up = 61;
    first_death = 35;
    final_day = first_day_hp + length(hospital_prevalence_data);
    global hospital_incidence_indices  = [i+first_day_hi for i in 1:length(hospital_incidence_data)];
    global hospital_prevalence_indices = [i+first_day_hp for i in 1:length(hospital_prevalence_data)];
    global icu_prevalence_indices      = [i+first_day_up for i in 1:length(icu_prevalence_data)];
    global death_indices               = [i+first_death for i in 1:length(death_data)];
    global initial_population = 66470000.0;
    global region_gl = region;
    global initial_control_dates = [10.,58.,84.,final_day-42.,final_day];

  else throw(DomainError("This is not a valid region"))
  end
  # Initial conditions
  final_prediction_day = 211;
  prediction_length = 225;
  global log_initial_infectious = log(0.1);
  global initial_state = @SVector [initial_population-exp(log_initial_infectious),(1-pA)*exp(log_initial_infectious),
                                   0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,pA*exp(log_initial_infectious),0.,0.,0.,0.,0.,0.,initial_population,0.];
  global parameters = [initial_rates,initial_probabilities,initial_transmission_rates];

  # MCMC
  number_of_samples = 50000
  # # do this for an initial run to get a variance covariance matrix
  # output = random_walk(delayed_ode,
  #                      number_of_samples,
  #                      [0.67,0.2,0.2,0.2,2.0,10.0,10.0,10.0,10.0,0.5,0.5,0.5,0.5],
  #                      0.001)
  # covariance_matrix = cov(output[Int(number_of_samples/2):end,:]);
  # initial_value = vec(mean(output[Int(number_of_samples/2):end,:],dims=1));
  # output = random_walk(delayed_ode,
  #                      number_of_samples,
  #                      initial_value,
  #                      0.33,
  #                      covariance_matrix,
  #                      1)
  # covariance_matrix = cov(output[Int(number_of_samples/2):end,:]);
  # initial_value = vec(mean(output[Int(number_of_samples/2):end,:],dims=1));
  # posterior_samples = random_walk(delayed_ode,
  #                                 number_of_samples,
  #                                 initial_value,
  #                                 0.4,
  #                                 covariance_matrix,
  #                                 1)
  # writedlm(string("$region","_output_test.csv"),posterior_samples,',')

  # do this once you have a covariance matrix for decent posteriors
  output = readdlm(string("$region","_output_test.csv"),',');
  covariance_matrix = cov(output[Int(size(output)[1]/2):end,:]);
  initial_value = vec(mean(output[Int(size(output)[1]/2):end,:],dims=1));
  posterior_samples = random_walk(delayed_ode,
                                  number_of_samples,
                                  initial_value,
                                  0.4,
                                  covariance_matrix,
                                  1)

  writedlm(string("$region","_output_test.csv"),posterior_samples,',')

  using MCMCChains
  chain = Chains(posterior_samples,["β₁","β₂","β₃","β₄","log(J₀)","σ_hi","σ_hp",
                                    "σ_up","σ_di","rCM","rHR","pC","pT"]);
  savefig(plot(chain),string("$region","_chain_test.png"))

  # make predictions
  posterior_samples = readdlm(string("$region","_output_test.csv"),',');
  println("Calculating predictive posterior distribution")
  number_of_samples = size(posterior_samples)[1];
  number_of_days = prediction_length+1;
  predicted_states = zeros(number_of_samples,21,number_of_days);
  rIH = 0.4;

  for (sample_index, sample) in enumerate(eachrow(posterior_samples))
    predicted_states[sample_index,:,:] = solve_ode(sample,prediction_length);
    if sample_index%(number_of_samples/10)==0
      println("Progress: ",100*sample_index/number_of_samples,'%')
    end
  end
  # posterior hospital incidence
  prediction = zeros(4,number_of_days,number_of_samples)
  for day in 10:number_of_days-5
    r = rIH*predicted_states[:,6,day]./(posterior_samples[:,6].-1)
    p = 1 ./ posterior_samples[:,6]
    for index in 1:number_of_samples
      prediction[1,day,index] = rand(NegativeBinomial(r[index],p[index]))
    end
    r = vec(sum(predicted_states[:,[8,9,10,11,12,21],day],dims=2)./(posterior_samples[:,7].-1))
    p = 1 ./ posterior_samples[:,7]
    for index in 1:number_of_samples
      prediction[2,day,index] = rand(NegativeBinomial(r[index],p[index]))
    end
    r = vec(sum(predicted_states[:,[10,11],day],dims=2)./(posterior_samples[:,8].-1))
    p = 1 ./ posterior_samples[:,8]
    for index in 1:number_of_samples
      prediction[3,day,index] = rand(NegativeBinomial(r[index],p[index]))
    end
    if region_gl in ["SC","WA"]
      r = (predicted_states[:,21,day].*initial_rates[11] .+ predicted_states[:,10,day].*initial_rates[6]*3)./(posterior_samples[:,9].-1)
    else
      r = (predicted_states[:,21,day].*initial_rates[11] .+ predicted_states[:,10,day].*initial_rates[6])./(posterior_samples[:,9].-1)
    end
    p = 1 ./ posterior_samples[:,9]
    for index in 1:number_of_samples
      prediction[4,day,index] = rand(NegativeBinomial(r[index],p[index]))
    end
  end

  writedlm(string("$region","_hi_nbinom_output_test.csv"),transpose(prediction[1,10:number_of_days-5,:]),',')
  writedlm(string("$region","_hp_nbinom_output_test.csv"),transpose(prediction[2,10:number_of_days-5,:]),',')
  writedlm(string("$region","_up_nbinom_output_test.csv"),transpose(prediction[3,10:number_of_days-5,:]),',')
  writedlm(string("$region","_di_nbinom_output_test.csv"),transpose(prediction[4,10:number_of_days-5,:]),',')

  # compute growth rate
  hp_growth_rate = transpose((log.(vec(sum(predicted_states[:,[8,9,10,11,12,21],final_prediction_day],dims=2)./(posterior_samples[:,7].-1))) .-
                              log.(vec(sum(predicted_states[:,[8,9,10,11,12,21],final_prediction_day-7],dims=2)./(posterior_samples[:,7].-1)))) ./ 7)
  hp_growth_rate = replace(hp_growth_rate, NaN => -Inf);
  writedlm(string("$region","_growth_rate_hp_test.csv"),transpose(quantile(vec(hp_growth_rate),0.05:0.05:0.95)),',');

  # compute r values
  all_r_values = posterior_samples[:,4] .* (predicted_states[:,1,final_prediction_day] ./ initial_population);
  writedlm(string("$region","_R_value_adjusted.csv"),transpose(quantile(all_r_values,0.05:0.05:0.95)*3.62),',');
  # store 90% quantiles for figures
  r_values = transpose(quantile(all_r_values,[0.05,0.5,0.95])*3.62)

  # compute prediction quantiles
  hi_quantiles = [quantile(prediction[1,i,:],j) for i in final_prediction_day-20:final_prediction_day,
                                                    j in 0.05:0.05:0.95];
  hp_quantiles = [quantile(prediction[2,i,:],j) for i in final_prediction_day-20:final_prediction_day,
                                                    j in 0.05:0.05:0.95];
  up_quantiles = [quantile(prediction[3,i,:],j) for i in final_prediction_day-20:final_prediction_day,
                                                    j in 0.05:0.05:0.95];
  di_quantiles = [quantile(prediction[4,i,:],j) for i in final_prediction_day-20:final_prediction_day,
                                                    j in 0.05:0.05:0.95];
  writedlm(string("$region","_posterior_predicted_quantiles_test.csv"),[hi_quantiles;
                                                                       hp_quantiles;
                                                                       up_quantiles;
                                                                       di_quantiles],',');

  # hospital incidence
  lower  = [quantile(prediction[1,i,:],0.025) for i in 1:prediction_length-6];
  upper = [quantile(prediction[1,i,:],0.975) for i in 1:prediction_length-6];
  median = [quantile(prediction[1,i,:],0.5) for i in 1:prediction_length-6];
  plot(1:length(upper),upper,color="#20948B",linestyle=:dash,alpha=0.4,legend=false)
  plot!(1:length(lower),lower,color="#20948B",linestyle=:dash,alpha=0.4)
  plot!(1:length(median),median,color="#20948B",alpha=0.4)
  plot!(first_day_hi:length(hospital_incidence_data)+(first_day_hi-1),hospital_incidence_data, seriestype=:scatter, color="black",markersize=3)
  savefig(string("$region","_posterior_hospital_incidence_test.pdf"))

  # hospital prevalence
  lower = [quantile(prediction[2,i,:],0.025) for i in 1:prediction_length-6];
  upper = [quantile(prediction[2,i,:],0.975) for i in 1:prediction_length-6];
  median = [quantile(prediction[2,i,:],0.5) for i in 1:prediction_length-6];
  plot(1:length(upper),upper,color="#20948B",linestyle=:dash,alpha=0.4,legend=false)
  plot!(1:length(lower),lower,color="#20948B",linestyle=:dash,alpha=0.4)
  plot!(1:length(median),median,color="#20948B",alpha=0.4)
  plot!(first_day_hp:length(hospital_prevalence_data)+(first_day_hp-1),hospital_prevalence_data, seriestype=:scatter, color="black",markersize=3)
  savefig(string("$region","_posterior_hospital_prevalence_test.pdf"))

  # icu prevalence
  lower = [quantile(prediction[3,i,:],0.025) for i in 1:prediction_length-6];
  upper = [quantile(prediction[3,i,:],0.975) for i in 1:prediction_length-6];
  median = [quantile(prediction[3,i,:],0.5) for i in 1:prediction_length-6];
  plot(1:length(upper),upper,color="#20948B",linestyle=:dash,alpha=0.4,legend=false)
  plot!(1:length(lower),lower,color="#20948B",linestyle=:dash,alpha=0.4)
  plot!(1:length(median),median,color="#20948B",alpha=0.4)
  plot!(first_day_up:length(icu_prevalence_data)+(first_day_up-1),icu_prevalence_data, seriestype=:scatter, color="black",markersize=3)
  savefig(string("$region","_posterior_icu_prevalence_test.pdf"))

  # death incidence
  lower = [quantile(prediction[4,i,:],0.025) for i in 1:prediction_length-6];
  upper = [quantile(prediction[4,i,:],0.975) for i in 1:prediction_length-6];
  median = [quantile(prediction[4,i,:],0.5) for i in 1:prediction_length-6];
  plot(1:length(upper),upper,color="#20948B",linestyle=:dash,alpha=0.4,legend=false)
  plot!(1:length(lower),lower,color="#20948B",linestyle=:dash,alpha=0.4)
  plot!(1:length(median),median,color="#20948B",alpha=0.4)
  plot!(first_death:length(death_data)+(first_death-1),death_data, seriestype=:scatter, color="black",markersize=3)
  savefig(string("$region","_posterior_death_test.pdf"))
end
