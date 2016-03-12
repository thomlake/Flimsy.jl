using JSON

function get_user_config()
    config_file_name = joinpath(pwd(), "config.json")
    for arg in ARGS
        if startswith(arg, "flimsy_config=")
            config_file_name = last(split(arg, "="; limit=2))
            break
        end
    end
    if isfile(config_file_name)
        println("Flimsy Info: found config file => ", config_file_name)
        return JSON.parsefile(config_file_name)
    else
        return Dict{ASCIIString,Any}()
    end
end

const FLIMSY_CONFIG = get_user_config()

# Setup default configuration
FLIMSY_CONFIG["always_check_bounds"] = get(FLIMSY_CONFIG, "always_check_bounds", false)
