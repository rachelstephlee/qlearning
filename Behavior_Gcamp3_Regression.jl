########################## IMPORTS ##################################
using MixedModels
using CSV
using Distributions
using Statistics
using StatsBase
using DataFrames
# functions for setting up data

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

##### prep for regression #####
# function prepares regression by: 1. correctly pull out timelocked data to float form 2. normalize q values
# 3. create variables for action, Q_vals, and interaction
function prepare_regression(df_src, timelock, q_vals_type)
    df_src[TIMELOCK] = format_vec.(df_src[TIMELOCK])
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

format_vec = str_vec->[parse(Float64, ss) for ss in split(str_vec[2:(end-1)])]


########################## IMPORT DATA ##################################

RECORDLOC = "DMS_CB"
Q_VALS_TYPE = :Q_ch_diff
TIMELOCK = :g_lp

# import data

df_corr_all = CSV.read("data/int_pc_qvals_gcamp_dms.csv", index_col = 0) # .dropna(subset=['Q_dir_diff'])



df_src = df_corr_all[(df_corr_all[:RecordLoc] .== RECORDLOC) , :]





########################## PREPARE REGRESSION ##################################


df = prepare_regression(df_src, TIMELOCK, Q_VALS_TYPE)



########################## REGRESSION ##################################

 # create interaction term
Action_float = ones(Float64, length(df.Action))
Action_float[df.Action .== "Ips"] .= 0
df.Action_float = Action_float
df.Interact = df.Action_float .* df[Q_VALS_TYPE]

fits = Array{Any}(nothing, 45)
coefs = Array{Any}(nothing, 45)
pvals = Array{Any}(nothing, (45))
df_fits = DataFrame(Estimate = Float64[], StdError = Float64[], Pval = Float64[], Variable = String[],
                        Time = Float64[])
for i in 1:45
    lhs = Symbol("x", i)
    formula_i = @eval @formula($lhs ~ 1 + Action_float + Q_ch_diff + Interact + (1 + Action_float + Q_ch_diff + Interact | MouseID))
    fits[i] = fit(LinearMixedModel, formula_i, df)
    idx = 1 + (i - 1) * 4: 1 + (i - 1) * 4 + 3
    co = coef(fits[i])
    se = stderror(fits[i]) # assuming this is calculating standard error
    z = co ./ se
    pval = ccdf.(Chisq(1), abs2.(z))

    df_i = DataFrame()
    df_i.Estimate = co
    df_i.StdError = se
    df_i.Pval = pval
    df_i.Variable = ["Intercept", "Action:Con", string(Q_VALS_TYPE), "Interact"]
    df_i.Time = ones(4) .* (-1 + 3/45 * (i - 1))

    # df_fits_old = copy(df_fits)
    # foo = vcat(df_fits, df_i)
    # df_fits = deepcopy(foo)
    append!(df_fits, df_i)
end

########################## SAVING DATAFRAMES ##################################

df_fits.timelocked = TIMELOCK
df_fits.RecordLoc = RECORDLOC

# CSV.write("data/" * src * "/Julia_lever_np.csv", df_fits) # now go correct first

########################## PLOT TO CHECK ##################################
using Plots

########################## P-VALUE CORRECTION ##################################

using MultipleTesting
# using CSV
# using DataFrames



for var = unique(df_fits[:Variable])
    println(var)
    pvals = convert(Array{Float64,1}, df_fits[df_fits[:Variable] .== var, :Pval])
    df_fits[df_fits[:Variable] .== var, :Pval] = adjust(PValues(pvals),MultipleTesting.BenjaminiHochberg())

end

CSV.write("data/" * RECORDLOC * "/Julia_lever_dir2_corrected.csv", df_fits)
