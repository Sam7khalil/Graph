import './style.css'
import _cat from './cat.json'
import _bg from './boardgames_100.json'
import * as go from 'gojs'


interface CatItem {
  title: string
  id: Number,
  main_categories: {
    id: Number,
    name: string,
  }[],
  poll_results: {
    votes: Number,
    percent: Number,
    winner: Boolean,
    category: {
      id: Number,
      name: string,
    },
  }[],
}


interface BoardGame {
  id: number;
  title: string;
  year: number;
  rank: number;
  minplayers: number;
  maxplayers: number;
  minplaytime: number;
  maxplaytime: number;
  minage: number;
  rating: {
    rating: number;
    num_of_reviews: number;
  };
  recommendations: {
    fans_liked: number[];
  };
  types: {
    categories: {
      id: number;
      name: string;
    }[];
    mechanics: {
      id: number;
      name: string;
    }[];
  };
  credit: {
    designer: {
      id: number;
      name: string;
    }[];
  };
}

const cat: CatItem[] = _cat
const games: BoardGame[] = _bg

const $ = go.GraphObject.make;

const G =
  $(go.Diagram, "app", {
    initialAutoScale: go.Diagram.Uniform,  // an initial automatic zoom-to-fit
    contentAlignment: go.Spot.Center,  // align document to the center of the viewport
    layout:
      $(go.ForceDirectedLayout,  // automatically spread nodes apart
        { maxIterations: 200, defaultSpringLength: 30, defaultElectricalCharge: 100 })
  })

G.nodeTemplate = $(go.Node, "Auto",  // the whole node panel
  $(go.Shape, "RoundedRectangle",
    new go.Binding("fill", "color"),
  ),  // the border
  $(go.TextBlock,
    new go.Binding("text", "title"))
)

G.linkTemplate = $(go.Link,  // the whole link panel
  $(go.Shape),  // the link shape, default black stroke
  $(go.Shape,  // the arrowhead
    { toArrow: "Standard" }),
)

const gameIds = new Set(games.map(g => g.id))
const colors = {
  "Strategy Games" : "red",
  "Family Games" : "blue",
  "Thematic Games" : "green",
  "Customizable Games" : "yellow",
}

const nodes = [...games.map((g, i) => ({
  key: g.id,
  title: g.title,
  color: colors[cat[i].main_categories[0].name]!,
}))]

const links = [...games.map(g => (g.recommendations.fans_liked.filter(f => gameIds.has(f)).map(f => ({
  from: g.id,
  to: f,
}))))].flat()

G.model = new go.GraphLinksModel(nodes, links)