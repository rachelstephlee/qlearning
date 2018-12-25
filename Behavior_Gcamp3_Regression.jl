########################## IMPORTS ##################################
using Pandas
using MixedModels
using Pkg

src = "DMS_CB"
Q_vals = "Q_ch_diff"
timelock = "g_np"

# import data

df_corr_all = read_csv("data/int_pc_qvals_gcamp_dms (1).csv", index_col = 0) # .dropna(subset=['Q_dir_diff'])


df_trials = read_csv("data/int_pc_qvals.csv", index_col = 0)


# update the gcamp values from strings --> arrays of floats
df_src = df_corr_all

format_vec = str_vec->[parse(Float64, ss) for ss in split(str_vec[2:(end-1)])]
df_src["g_np"] = format_vec.(df_src["g_np"])
df_src["g_lp"] = format_vec.(df_src["g_lp"])
df_src["g_choice"] = format_vec.(df_src["g_choice"])
df_src["g_reward"] = format_vec.(df_src["g_reward"])


# df_corr = df_src[df_src["RecordLoc"] == "DMS"]
df_corr = reset_index(df_src[df_src["RecordLoc"] == src])
# df_corr = pd.DataFrame(df_corr_all[df_corr_all['RecordLoc'] == 'DMS_CB']).reset_index()

########################## NORMALIZE ##################################

function normalize(x, mu, std)
    return (x .- mu) ./ std
end

println("Using Q_vals: "* Q_vals)


for mouse_id in unique(df_corr["MouseID"])

    mice_q = copy(values(df_corr[df_corr["MouseID"] == mouse_id][Q_vals]));
    mice_mu = mean(mice_q[.!isnan.(mice_q)]);
    mice_std = std(mice_q[.!isnan.(mice_q)]);

    loc(df_corr)[df_corr["MouseID"] == mouse_id, Q_vals] = normalize(mice_q, mice_mu, mice_std)

end


########################## PREP FOR REGRESSION ##################################


println("Using Timelocked GCaMP6f: " * string(timelock))
df_corr_r = drop(df_corr, ["g_np", "g_lp", "g_choice", "g_reward", "Stay/Leave"], axis = 1)

gcamp_temp = DataFrame([values(x) for x in df_corr[timelock]], columns = ["G" * string(x) for x in 1:45]) #update


df_gcamp_reg = dropna(join(df_corr_r, gcamp_temp))

to_csv(df_gcamp_reg, "df_gcamp_reg_np_cb" * ".csv")

########################## RESTART JULIA: REGRESSION SET UP ##################################

using CSV
using Distributions
using Statistics
using MixedModels
using StatsBase
using DataFrames

src = "DMS_CB"
Q_vals = "Q_ch_diff"
timelock = "g_np"


df = CSV.read("df_gcamp_reg_np_cb" * ".csv")

########################## REGRESSION ##################################

 # create interaction term
Action_float = ones(Float64, length(df.Action))
Action_float[df.Action .== "Ips"] .= 0
df.Action_float = Action_float
df.Interact = df.Action_float .* df.Q_ch_diff
println("Make sure interaction is correct. need to change it by hand! ")

fits = Array{Any}(nothing, 45)
coefs = Array{Any}(nothing, 45)
pvals = Array{Any}(nothing, (45))
df_fits = DataFrame(Estimate = Float64[], StdError = Float64[], Pval = Float64[], Variable = String[],
                        Time = Float64[])
for i in 1:45
    lhs = Symbol("G", i)
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
    df_i.Variable = ["Intercept", "Action:Con", Q_vals, "Interact"]
    df_i.Time = ones(4) .* (-1 + 3/45 * (i - 1))

    # df_fits_old = copy(df_fits)
    # foo = vcat(df_fits, df_i)
    # df_fits = deepcopy(foo)
    append!(df_fits, df_i)
end

# so fast <3 <3 <3 Julia JIT!!!!
########################## SAVING DATAFRAMES ##################################

df_fits.timelocked = timelock
df_fits.RecordLoc = src
CSV.write("data/" * src * "/Julia_lever_np.csv", df_fits)

########################## PLOT TO CHECK ##################################
using Plots
plot([coef[1] for coef in coefs])
