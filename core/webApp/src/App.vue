<template>
  <v-app>
    <v-toolbar app>
      <v-toolbar-title>youShakespeare</v-toolbar-title>
    </v-toolbar>
    <main>
      <v-content>
        <v-container fluid>
          <v-container fluid>
            <div class='pb-2 pt-2'>
              <h4>Generate Shakespeare text by using AI</h4>
              <h6>Type something below and generate some shakespeare!</h6>
            </div>
          </v-container>
          <v-layout row wrap>
            <v-flex xs12>
              <v-container>
                <v-layout row nowrap>
                  <v-flex>
                    <v-text-field label="Initial text" v-model="inputVal" :rules="textRules"></v-text-field>
                  </v-flex>
                  <v-btn @click="generateText()" :disabled="loading" :loading="loading">SEND</v-btn>
                </v-layout>
                <v-layout row nowrap align-center>
                  <v-flex>
                    How many characters? The more the slower.
                    <v-slider v-model="nText" thumb-label min=100 max=2000 step=100 snap></v-slider>
                  </v-flex>
                  <p>{{nText}} </p>
                </v-layout>
              </v-container>
            </v-flex>
            <v-flex xs12>
              <div v-if="generated.length > 0" class='pt-2 pb-2 text-xs-center'>
                <h6>Result </h6>
                <hr/>
              </div>
              <div v-else>
                <h6 class='text-xs-center center-absolute'> Nothing here! Generate something </h6>
              </div>
              <v-container>
              <pre>{{generated}}</pre>
              </v-container>
            </v-flex>
          </v-layout>
        </v-container>
      </v-content>
    </main>
    <v-footer class="pa-3">
    <v-spacer></v-spacer>
    <div>© Francesco Saverio Zuppichini {{ new Date().getFullYear() }}</div>
  </v-footer>
  </v-app>
</template>

<script>
  import HelloWorld from './components/HelloWorld'
  import axios from 'axios'
  
  export default {
    name: 'app',
    components: {
      HelloWorld
    },
    data() {
      return {
        generated: '',
        inputVal: '',
        loading: false,
        nText: 0,
        textRules: [
          (v) => !!v || 'Type something!',
        ],
      }
    },
    methods: {
      generateText() {
        if(this.inputVal.length < 1) return

        this.loading = true

        axios.get(`api/generate/${this.inputVal}/${this.nText}`)
          .then(({
            data
          }) => {
            this.loading = false
            this.generated = data.result
          })
          .catch((err) => {
            console.log(err)
            this.loading = false
          })
      }
    }
  }
</script>

<style>
  .center-absolute {
    position: absolute;
    top: 50%;
    width: 100%;
  }
</style>
