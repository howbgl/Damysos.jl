using ArgParse
function parse_cmdargs()
      s = ArgParseSettings()
      @add_arg_table! s begin
            "parameterset"
            help = "Index of the parameter set, int from 1 to 10"
            arg_type = Int
            required = true
      end
      return parse_args(s)
end

const cmdargs = parse_cmdargs()

@show cmdargs["parameterset"]