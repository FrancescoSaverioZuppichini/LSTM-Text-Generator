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
               <v-container fluid>
              <v-layout row nowrap>
                <v-flex >
                    <v-text-field label="Initial text" v-model="inputVal"></v-text-field>
                </v-flex>
                  <v-btn @click="generateText()"  :disabled="loading" :loading="loading">SEND</v-btn>
              </v-layout>
              </v-container>
            </v-flex>
            <v-flex xs12>
              <h6 v-if="generated.length > 0">Result </h6>
              <pre>
              {{generated}}
              </pre>
            </v-flex>
          </v-layout>
        </v-container>
      </v-content>
    </main>
    <v-footer app></v-footer>
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
      }
    },
    methods: {
      generateText() {
        this.loading = true
        axios.get(`api/generate/${this.inputVal}/1000`)
          .then(({
            data
          }) => {
            this.loading = false
            this.generated = data.result
          })
          .catch(() => {
            this.loading = false
          })
      }
    }
  }
</script>

<style>
  
</style>
