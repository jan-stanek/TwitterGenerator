/**
 * Created by Admin on 04.12.2016.
 */
public class LstmGenerate {
    public static void main(String[] args) {

    }


    public char convertIndexToCharacter(int idx) {
        return indexToCharArray[idx];
    }

    public int convertCharacterToIndex(char c) {
        return charToIndexMap.get(c);
    }

    public char getRandomCharacter() {
        return indexToCharArray[(int) (rng.nextDouble() * validChars.size())];
    }

}
