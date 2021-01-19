using Agents, GLMakie, AbstractPlotting

mutable struct Bird <: AbstractAgent
    id::Int
    pos::NTuple{3,Float64}
    vel::NTuple{3,Float64}
    speed::Float64
    cohere_factor::Float64
    separation::Float64
    separate_factor::Float64
    match_factor::Float64
    visual_distance::Float64
end

function initialize_model(;
    n_birds = 100,
    speed = 1.0,
    cohere_factor = 0.25,
    separation = 4.0,
    separate_factor = 0.25,
    match_factor = 0.01,
    visual_distance = 5.0,
    extent = (100, 100, 100),
    spacing = visual_distance / 1.5,
)
    space3d = ContinuousSpace(extent, spacing)
    model = ABM(Bird, space3d, scheduler = random_activation)
    for _ in 1:n_birds
        vel = Tuple(rand(3) * 2 .- 1)
        add_agent!(
            model,
            vel,
            speed,
            cohere_factor,
            separation,
            separate_factor,
            match_factor,
            visual_distance,
        )
    end
    return model
end

function agent_step!(bird, model)
    ## Obtain the ids of neighbors within the bird's visual distance
    neighbor_ids = nearby_ids(bird, model, bird.visual_distance)
    N = 0
    match = separate = cohere = (0.0, 0.0, 0.0)
    ## Calculate behaviour properties based on neighbors
    for id in neighbor_ids
        N += 1
        neighbor = model[id].pos
        heading = neighbor .- bird.pos

        ## `cohere` computes the average position of neighboring birds
        cohere = cohere .+ heading
        if edistance(bird.pos, neighbor, model) < bird.separation
        ## `separate` repels the bird away from neighboring birds
            separate = separate .- heading
        end
        ## `match` computes the average trajectory of neighboring birds
        match = match .+ model[id].vel
    end
    N = max(N, 1)
    ## Normalise results based on model input and neighbor count
    cohere = cohere ./ N .* bird.cohere_factor
    separate = separate ./ N .* bird.separate_factor
    match = match ./ N .* bird.match_factor
    ## Compute velocity based on rules defined above
    bird.vel = (bird.vel .+ cohere .+ separate .+ match) ./ 2
    bird.vel = bird.vel ./ norm(bird.vel)
    ## Move bird according to new velocity and speed
    move_agent!(bird, model, bird.speed)
end


function abm_plot(model; res=(800, 600), kwargs...)
    scene = Scene(; resolution=res)
    pos = abm_plot!(scene, model)
    return pos
end

function modellims(model)
    if model.space isa Agents.ContinuousSpace
        e = model.space.extent
        o = zero.(e) .+ 0.5
    elseif model.space isa Agents.DiscreteSpace
        e = size(model.space.s) .+ 1
        o = zero.(e)
    end
    return o, e
end

function abm_plot!(scene, model,
    ac = "#ff0000",
    as = 1,
    am = Sphere(Point3f0(0), 1f0),
    scheduler = model.scheduler,
    offset = nothing,
    equalaspect = true,
    scatterkwargs = NamedTuple(),
)
    o, e = modellims(model)
    @assert typeof(model.space) <: Union{Agents.ContinuousSpace, Agents.DiscreteSpace}

    ids = scheduler(model)
    colors  = ac isa Function ? Observable(to_color.([ac(model[i]) for i ∈ ids])) : to_color(ac)
    sizes   = as isa Function ? Observable([as(model[i]) for i ∈ ids]) : as
    markers = am isa Function ? Observable([am(model[i]) for i ∈ ids]) : am
    if offset == nothing
        pos = Observable([model[i].pos for i ∈ ids])
    else
        pos = Observable([model[i].pos .+ offset(model[i]) for i ∈ ids])
    end
    
    meshscatter!(
        scene,
        pos;
        color = colors, markersize = sizes, marker = markers,
        scatterkwargs...
    )
    xlims!(scene, o[1], e[1])
    ylims!(scene, o[2], e[2])
    zlims!(scene, o[3], e[3])
    equalaspect && (scene.aspect = AxisAspect(1))
    return pos, colors, sizes, markers
end

model = initialize_model()
abm_plot(model, as=0.2)