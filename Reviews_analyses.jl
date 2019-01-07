
# Latency Analyses (with Regression)
using CSV
using Distributions
using Statistics
using MixedModels
using StatsBase
using DataFrames

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
############################################
# EARLY VS LATE BLOCKS
############################################


RECORDLOC = "DMS"
BLOCK = "late"   # early or late
Q_VALS_TYPE = :Q_ch_diff
TIMELOCK = :g_lp

df_src = CSV.read("data/" * RECORDLOC * "/df_" * BLOCK * ".csv")


##### prep for regression #####
# formats the vector so that it's floats and not strings (for the gcamp)
format_vec = str_vec->[parse(Float64, ss) for ss in split(str_vec[2:(end-1)])]

df_src[TIMELOCK] = format_vec.(df_src[TIMELOCK])
df_src = normalize_q_vals(df_src, Q_VALS_TYPE)


df_reg = DataFrame(df_src[TIMELOCK][1]')
for x in df_src[TIMELOCK][2:end]
    push!(df_reg, Array(values(x)'))
end

# add relevant variables for regression
df_reg[:Action] = df_src[:Action]
df_reg[Q_VALS_TYPE] = df_src[Q_VALS_TYPE]
df_reg[:MouseID] = df_src[:MouseID]

Action_float = ones(Float64, length(df_reg.Action))
Action_float[df_reg.Action .== "Ips"] .= 0
df_reg.Action_float = Action_float
df_reg.Interact = df_reg[:Action_float] .* df_reg[Q_VALS_TYPE]


### perform regression
df_fits = DataFrame(Estimate = Float64[], StdError = Float64[], Pval = Float64[], Variable = String[],
                       Time = Float64[], Timelock = String[], RecordLoc = String[])
for i in 1:45
   lhs = Symbol("x", i)
   formula_i = @eval @formula($lhs ~ 1 + Action_float + $Q_VALS_TYPE + Interact + (1 + Action_float + Q_ch_diff + Interact | MouseID))
   fits_i = fit(LinearMixedModel, formula_i, df_reg)
   idx = 1 + (i - 1) * 4: 1 + (i - 1) * 4 + 3
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

df_fits = CSV.read("data/" * RECORDLOC * "/df_" * BLOCK * "_reg.csv")

### correcting p-values

using MultipleTesting


RECORDLOC = "DMS"
BLOCK = "early"

for var = unique(df_fits[:Variable])
    println(var)
    pvals = convert(Array{Float64,1}, df_fits[df_fits[:Variable] .== var, :Pval])
    df_fits[df_fits[:Variable] .== var, :Pval] = adjust(PValues(pvals),MultipleTesting.BenjaminiHochberg())

end



#### saving

CSV.write("data/" * RECORDLOC * "/df_" * BLOCK * "_reg_corrected.csv", df_fits)

#################################
# LATENCY
#################################
# take mouse 4 out

# Lmer(latency ~ 1 + Qchdiff + presLat + (1 + Qchdiff + Preslat | mouseid), …)

df = CSV.read("data/latency_q_val_gcamp.csv")
df = normalize_q_vals(df)
df[:l_Latency_press] = log.(df[:Latency_press])
df[:l_Latency_choice] = log.(df[:Latency_choice])

#drop missing values
df[:StayVSLeave] = coalesce.(df[:StayVSLeave], "na")


# Lmer(latency ~ 1 + Qchdiff + presLat + (1 + Qchdiff + Preslat | mouseid), …)
df_DMS =  df[(df[:StayVSLeave] .== "stay"), :]
df_DMS_stay = df[(df[:RecordLoc].== "DMS") .& (df[:StayVSLeave] .== "stay"), :]
df_DMS_leave = df[(df[:RecordLoc].== "DMS") .& (df[:StayVSLeave] .== "leave"), :]


formula_i = @eval @formula(l_Latency_choice ~ 1 + Q_ch_diff + l_Latency_press + (1 + Q_ch_diff + l_Latency_press | MouseID))

latency_fits_stay = fit(LinearMixedModel, formula_i, df_DMS_stay)
latency_fits_leave = fit(LinearMixedModel, formula_i, df_DMS_leave)


df_DMS_CB = df[df[:RecordLoc].== "DMS_CB", :]
df_DMS_CB_stay = df[(df[:RecordLoc].== "DMS_CB") .& (df[:StayVSLeave] .== "stay"), :]
df_DMS_CB_leave = df[(df[:RecordLoc].== "DMS_CB") .& (df[:StayVSLeave] .== "leave"), :]


formula_i = @eval @formula(l_Latency_choice ~ 1 + Q_ch_diff + l_Latency_press + (1 + Q_ch_diff + l_Latency_press | MouseID))
latency_fits = fit(LinearMixedModel, formula_i, df_DMS_CB)

latency_fits_stay = fit(LinearMixedModel, formula_i, df_DMS_CB_stay)
latency_fits_leave = fit(LinearMixedModel, formula_i, df_DMS_CB_leave)
