import React from 'react'
import etsiinf from './LOGOTIPO_ETSIINF.png';
import axios from 'axios';
import './Template.css'
import { Button } from '@mui/material';
import {TextField} from '@mui/material'

class Inicio extends React.Component {
    constructor(props) {
        super(props)
        this.state = {

          imagen: null,
          texto:"",
          emoticono:"",
        }
      }
    
    onChange = e => {
        this.setState({imagen:URL.createObjectURL(e.target.files[0])})
        console.log(this.state.imagen)
    }
    llamada = () => {
        let envio = this.state.texto.split("base64,")[1]
        let json = { "imagen" : envio}
        axios.post('http://127.0.0.1:5000/emojinet',json).
        then(res => {
            this.setState({emoticono: res.data.respuesta})
            console.log(res)
        })
        
    }
    //futuras funciones
    render() {
        //lo que se renderiza
        return (
            <>
     
                <div className="App">
                    <div className="Front">

                        <header className="App-header">
                        <img src={etsiinf} className="App-logo" alt="logo" />
                            <h1>EmojiNet</h1>

                             <TextField id="filled-basic" label="Insertar Base64" variant="filled"
                             onChange={(e) => this.setState({texto: e.target.value})}
                             />
                             <Button variant="contained" onClick={this.llamada.bind(this)}>Ejecutar</Button>
                             {this.state.emoticono ? this.state.emoticono : "Escribe un base 64 y EJECUTA !"}
                             <img className="App-logo" src={this.state.texto  ? this.state.texto: "" }/>
                             

                        </header>
                    </div>
                </div >
            </>
        );
    }
}
export default Inicio