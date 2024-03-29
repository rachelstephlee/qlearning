
# Latency Analyses (with Regression)
using CSV
using Distributions
using Statistics
using MixedModels
using StatsBase
using DataFrames
using Plots
using MultipleTesting


function normalize(x, mu, std)
    return (x .- mu) ./ std
end

# normalizes the Q vals, default is choice difference
function normalize_q_vals(df, q_val)
    for mouse_id in unique(df[:MouseID])

        mice_q = copy(values(df[df[:MouseID] .== mouse_id,q_val]));
        mice_mu = mean(mice_q[.!isnan.(mice_q)]);
        mice_std = std(mice_q[.!isnan.(mice_q)]);

        df[df[:MouseID] .== mouse_id,:Q_ch_diff] = normalize(mice_q, mice_mu, mice_std)

    end
    return df

end

# formats the vector so that it's floats and not strings (for the gcamp)
format_vec = str_vec->[parse(Float64, ss) for ss in split(str_vec[2:(end-1)])]

##### prep for regression #####
# function prepares regression by: 1. correctly pull out timelocked data to float form 2. normalize q values
# 3. create variables for action, Q_vals, and interaction
function prepare_regression(df_src, timelock, q_vals_type)
    df_src[timelock] = format_vec.(df_src[timelock])
    df_src = normalize_q_vals(df_src, q_vals_type)


    df_reg = DataFrame(df_src[timelock][1]')
    for x in df_src[timelock][2:end]
        push!(df_reg, Array(values(x)'))
    end

    # add relevant variables for regression
    df_reg[:Action] = df_src[:Action]
    df_reg[q_vals_type] = df_src[q_vals_type]
    df_reg[:MouseID] = df_src[:MouseID]

    Action_float = ones(Float64, length(df_reg.Action))
    Action_float[df_reg.Action .== "Ips"] .= 0
    df_reg.Action_float = Action_float
    df_reg.Interact = df_reg[:Action_float] .* df_reg[q_vals_type]

    return df_reg
end


function correct_pvals(df_fits)

    for var = unique(df_fits[:Variable])
        println(var)
        pvals = convert(Array{Float64,1}, df_fits[df_fits[:Variable] .== var, :Pval])
        df_fits[df_fits[:Variable] .== var, :Pval] = adjust(PValues(pvals),MultipleTesting.BenjaminiHochberg())

    end

    return df_fits

end

############################################
# EARLY VS LATE BLOCKS
############################################


RECORDLOC = "DMS"
BLOCK = "late"   # early or late
Q_VALS_TYPE = :Q_ch_diff
TIMELOCK = :g_lp

df_src = CSV.read("data/" * RECORDLOC * "/df_" * BLOCK * ".csv")




df_reg = prepare_regression(df_src, TIMELOCK, Q_VALS_TYPE)

### perform regression
df_fits = DataFrame(Estimate = Float64[], StdError = Float64[], Pval = Float64[], Variable = String[],
                       Time = Float64[], Timelock = String[], RecordLoc = String[])
for i in 1:45
   lhs = Symbol("x", i)
   formula_i = @eval @formula($lhs ~ 1 + Action_float + $Q_VALS_TYPE + Interact + (1 + Action_float + $Q_VALS_TYPE + Interact | MouseID))
   fits_i = fit(LinearMixedModel, formula_i, df_reg)
   co = coef(fits_i)
   se = stderror(fits_i) # assuming this is calculating standard error
   z = co ./ se
   pval = ccdf.(Chisq(1), abs2.(z))

   df_i = DataFrame()
   df_i.Estimate = co
   df_i.StdError = se
   df_i.Pval = pval
   df_i.Variable = ["Intercept", "Action:Con", string(Q_VALS_TYPE), "Interact"]
   df_i.Time = ones(4) .* (-1 + 3/45 * (i - 1))
   df_i.Timelock = string(TIMELOCK)
   df_i.RecordLoc = RECORDLOC


   append!(df_fits, df_i)
end


### correcting p-values

df_fits = correct_pvals(df_fits)


#### saving

CSV.write("data/" * RECORDLOC * "/df_" * BLOCK * "_reg_corrected.csv", df_fits)

#################################
# LATENCY
#################################

function split_trials(df_src, recordloc)
    df_all =  df_src[(df_src[:RecordLoc] .== recordloc), :]
    df_stay = df_src[(df_src[:RecordLoc].== recordloc) .& (df_src[:StayVSLeave] .== "stay"), :]
    df_leave = df_src[(df_src[:RecordLoc].== recordloc) .& (df_src[:StayVSLeave] .== "leave"), :]

    return (df_all, df_stay, df_leave)
end

function pull_regression_coefs(df_fits, fit, recordloc, stayvsleave, var_names)
    co = coef(fit)
    se = stderror(fit) # assuming this is calculating standard error
    z = co ./ se
    pval = ccdf.(Chisq(1), abs2.(z))

    df_i = DataFrame()
    df_i.Estimate = co
    df_i.StdError = se
    df_i.Pval = pval
    df_i.Variable = var_names
    df_i.RecordLoc = recordloc
    df_i.StayVSLeave = stayvsleave


    return df_i
end


#############
# LATENCY: Adding it to the main regression
#############

RECORDLOC = "DMS"
Q_VALS_TYPE = :Q_ch_diff
TIMELOCK = :g_lp

df_corr_all = CSV.read("data/int_pc_qvals_gcamp_dms.csv", index_col = 0) # .dropna(subset=['Q_dir_diff'])



df_src = df_corr_all[(df_corr_all[:RecordLoc] .== RECORDLOC) , :]


df_reg = prepare_regression(df_src, TIMELOCK, Q_VALS_TYPE)

# add latency information to df_reg

df_latency = CSV.read("data/latency_qvals_gcamp.csv")


df_full = join(df_src, df_latency[(df_latency[:RecordLoc] .== RECORDLOC) , :], on = [:MouseID, :Trial, :RecordLoc, :Session], makeunique= true)
df_reg[:Latency_choice] = df_full[:Latency_choice]


### perform regression
df_fits = DataFrame(Estimate = Float64[], StdError = Float64[], Pval = Float64[], Variable = String[],
                       Time = Float64[], Timelock = String[], RecordLoc = String[])
for i in 1:45
   lhs = Symbol("x", i)
   formula_i = @eval @formula($lhs ~ 1 + Action_float + $Q_VALS_TYPE + Interact + Latency_choice + (1 + Action_float + $Q_VALS_TYPE + Interact  + Latency_choice  | MouseID))
   fits_i = fit(LinearMixedModel, formula_i, df_reg)
   co = coef(fits_i)
   se = stderror(fits_i) # assuming this is calculating standard error
   z = co ./ se
   pval = ccdf.(Chisq(1), abs2.(z))

   df_i = DataFrame()
   df_i.Estimate = co
   df_i.StdError = se
   df_i.Pval = pval
   df_i.Variable = ["Intercept", "Action:Con", string(Q_VALS_TYPE), "Interact", "Lat_choice"]
   df_i.Time = ones(length(df_i.Variable)) .* (-1 + 3/45 * (i - 1))
   df_i.Timelock = string(TIMELOCK)
   df_i.RecordLoc = RECORDLOC


   append!(df_fits, df_i)
end

# p value correcting



df_fits = correct_pvals(df_fits)

CSV.write("data/" * RECORDLOC * "/Julia_lever_latency_corrected.csv", df_fits)


#############
# LATENCY: Adding it to the main regression (CONTRA ONLY)
#############

RECORDLOC = "DMS_CB"
Q_VALS_TYPE = :Q_ch_diff
TIMELOCK = :g_lp

df_corr_all = CSV.read("data/int_pc_qvals_gcamp_dms (1).csv", index_col = 0) # .dropna(subset=['Q_dir_diff'])


# df_src = df_corr_all[df_corr_all[:RecordLoc] .== RECORDLOC, :]
# only checking contralateral choices!
print("Only Contra choices") #
df_src = df_corr_all[(df_corr_all[:RecordLoc] .== RECORDLOC) .& (df_corr_all[:Action] .== "Con") , :]


df_reg = prepare_regression(df_src, TIMELOCK, Q_VALS_TYPE)

# add latency information to df_reg

df_latency = CSV.read("data/latency_qvals_gcamp.csv")

print("Only Contra Choices") #
df_full = join(df_src, df_latency[(df_latency[:RecordLoc] .== RECORDLOC) .& (df_latency[:Action] .== "Con"), :], on = [:MouseID, :Trial, :RecordLoc, :Session], makeunique= true)
df_reg[:Latency_choice] = df_full[:Latency_choice]


### perform regression
df_fits = DataFrame(Estimate = Float64[], StdError = Float64[], Pval = Float64[], Variable = String[],
                       Time = Float64[], Timelock = String[], RecordLoc = String[])
for i in 1:45
   lhs = Symbol("x", i)
   formula_i = @eval @formula($lhs ~ 1 + $Q_VALS_TYPE +  Latency_choice + (1 +  $Q_VALS_TYPE + Latency_choice  | MouseID))
   fits_i = fit(LinearMixedModel, formula_i, df_reg)
   co = coef(fits_i)
   se = stderror(fits_i) # assuming this is calculating standard error
   z = co ./ se
   pval = ccdf.(Chisq(1), abs2.(z))

   df_i = DataFrame()
   df_i.Estimate = co
   df_i.StdError = se
   df_i.Pval = pval
   df_i.Variable = ["Intercept", string(Q_VALS_TYPE), "Lat_choice"]
   df_i.Time = ones(length(df_i.Variable)) .* (-1 + 3/45 * (i - 1))
   df_i.Timelock = string(TIMELOCK)
   df_i.RecordLoc = RECORDLOC


   append!(df_fits, df_i)
end

# p value correcting



df_fits = correct_pvals(df_fits)

CSV.write("data/" * RECORDLOC * "/Julia_lever_latency_corrected_CONTRA.csv", df_fits)
